#include "DeferredPipeline.h"
#include "GBufferFlow.h"
#include "LightingFlow.h"
#include "../RenderView.h"
#include "../shadow/ShadowFlow.h"
#include "../ui/UIFlow.h"
#include "../SceneCulling.h"
#include "gfx/GFXBuffer.h"
#include "gfx/GFXCommandBuffer.h"
#include "gfx/GFXDescriptorSet.h"
#include "gfx/GFXDevice.h"
#include "gfx/GFXQueue.h"
#include "gfx/GFXRenderPass.h"
#include "gfx/GFXTexture.h"
#include "gfx/GFXFramebuffer.h"
#include "platform/Application.h"

namespace cc {
namespace pipeline {
namespace {
#define TO_VEC3(dst, src, offset) \
    dst[offset] = src.x;          \
    dst[offset + 1] = src.y;      \
    dst[offset + 2] = src.z;
#define TO_VEC4(dst, src, offset) \
    dst[offset] = src.x;          \
    dst[offset + 1] = src.y;      \
    dst[offset + 2] = src.z;      \
    dst[offset + 3] = src.w;
} // namespace

#define INIT_GLOBAL_DESCSET_LAYOUT(info)                                    \
while (1) {                                                                 \
    globalDescriptorSetLayout.samplers[info::NAME] = info::LAYOUT;          \
    globalDescriptorSetLayout.bindings[info::BINDING] = info::DESCRIPTOR;   \
}

DeferredPipeline::DeferredPipeline()
: RenderPipeline() {

    INIT_GLOBAL_DESCSET_LAYOUT(SAMPLERGBUFFERALBEDOMAP);
    INIT_GLOBAL_DESCSET_LAYOUT(SAMPLERGBUFFERPOSITIONMAP);
    INIT_GLOBAL_DESCSET_LAYOUT(SAMPLERGBUFFEREMISSIVEMAP);
    INIT_GLOBAL_DESCSET_LAYOUT(SAMPLERGBUFFERNORMALMAP);

    localDescriptorSetLayout.blocks[UBODeferredLight::NAME] = UBODeferredLight::LAYOUT;
    localDescriptorSetLayout.bindings[UBODeferredLight::BINDING] = UBODeferredLight::DESCRIPTOR;

}

gfx::RenderPass *DeferredPipeline::getOrCreateRenderPass(gfx::ClearFlags clearFlags) {
    if (_renderPasses.find(clearFlags) != _renderPasses.end()) {
        return _renderPasses[clearFlags];
    }

    auto device = gfx::Device::getInstance();
    gfx::ColorAttachment colorAttachment;
    gfx::DepthStencilAttachment depthStencilAttachment;
    colorAttachment.format = device->getColorFormat();
    depthStencilAttachment.format = device->getDepthStencilFormat();

    if (!(clearFlags & gfx::ClearFlagBit::COLOR)) {
        if (clearFlags & static_cast<gfx::ClearFlagBit>(SKYBOX_FLAG)) {
            colorAttachment.loadOp = gfx::LoadOp::DISCARD;
        } else {
            colorAttachment.loadOp = gfx::LoadOp::LOAD;
            colorAttachment.beginLayout = gfx::TextureLayout::PRESENT_SRC;
        }
    }

    if (static_cast<gfx::ClearFlagBit>(clearFlags & gfx::ClearFlagBit::DEPTH_STENCIL) != gfx::ClearFlagBit::DEPTH_STENCIL) {
        if (!(clearFlags & gfx::ClearFlagBit::DEPTH)) depthStencilAttachment.depthLoadOp = gfx::LoadOp::LOAD;
        if (!(clearFlags & gfx::ClearFlagBit::STENCIL)) depthStencilAttachment.stencilLoadOp = gfx::LoadOp::LOAD;
        depthStencilAttachment.beginLayout = gfx::TextureLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    auto renderPass = device->createRenderPass({
        {colorAttachment},
        depthStencilAttachment,
    });
    _renderPasses[clearFlags] = renderPass;

    return renderPass;
}

void DeferredPipeline::setFog(uint fog) {
    _fog = GET_FOG(fog);
}

void DeferredPipeline::setAmbient(uint ambient) {
    _ambient = GET_AMBIENT(ambient);
}

void DeferredPipeline::setSkybox(uint skybox) {
    _skybox = GET_SKYBOX(skybox);
}

//void DeferredPipeline::setShadows(uint shadows) {
//    _shadows = GET_SHADOWS(shadows);
//}

bool DeferredPipeline::initialize(const RenderPipelineInfo &info) {
    RenderPipeline::initialize(info);

    if (_flows.size() == 0) {
        auto shadowFlow = CC_NEW(ShadowFlow);
        shadowFlow->initialize(ShadowFlow::getInitializeInfo());
        _flows.emplace_back(shadowFlow);

        auto gbufferFlow = CC_NEW(GBufferFlow);
        gbufferFlow->initialize(GBufferFlow::getInitializeInfo());
        _flows.emplace_back(gbufferFlow);

        auto lightingFlow = CC_NEW(LightingFlow);
        lightingFlow->initialize(LightingFlow::getInitializeInfo());
        _flows.emplace_back(lightingFlow);

        auto uiFlow = CC_NEW(UIFlow);
        uiFlow->initialize(UIFlow::getInitializeInfo());
        _flows.emplace_back(uiFlow);
    }
    _sphere = CC_NEW(Sphere);

    return true;
}

bool DeferredPipeline::activate() {
    if (!RenderPipeline::activate()) {
        CC_LOG_ERROR("RenderPipeline active failed.");
        return false;
    }

    if (!activeRenderer()) {
        CC_LOG_ERROR("DeferredPipeline startup failed!");
        return false;
    }

    return true;
}

void DeferredPipeline::render(const vector<RenderView *> &views) {
    _commandBuffers[0]->begin();
    for (const auto view : views) {
        sceneCulling(this, view);
        for (const auto flow : view->getFlows()) {
            flow->render(view);
        }
    }
    _commandBuffers[0]->end();
    _device->getQueue()->submit(_commandBuffers);
}

void DeferredPipeline::updateUBOs(RenderView *view) {
    updateUBO(view);
    const auto scene = view->getCamera()->getScene();
    const Light *mainLight = nullptr;
    if (scene->mainLightID) mainLight = scene->getMainLight();
    const auto shadowInfo = _shadows;
    auto *device = gfx::Device::getInstance();

    if (mainLight && shadowInfo->enabled && shadowInfo->getShadowType() == ShadowType::SHADOWMAP) {
        const auto node = mainLight->getNode();
        cc::Mat4 matShadowCamera;

        // light proj
        float x = 0.0f, y = 0.0f, farClamp = 0.0f;
        if (shadowInfo->autoAdapt) {
            Vec3 tmpCenter;
            getShadowWorldMatrix(_sphere, node->worldRotation, mainLight->direction, matShadowCamera, tmpCenter);

            const float radius = _sphere->radius;
            x = radius * shadowInfo->aspect;
            y = radius;

            const float halfFar = tmpCenter.distance(_sphere->center);
            farClamp = std::min(halfFar * COEFFICIENT_OF_EXPANSION, SHADOW_CAMERA_MAX_FAR);
        } else {
            matShadowCamera = mainLight->getNode()->worldMatrix;

            x = shadowInfo->orthoSize * shadowInfo->aspect;
            y = shadowInfo->orthoSize;

            farClamp = shadowInfo->farValue;
        }

        const auto matShadowView = matShadowCamera.getInversed();

        Mat4 matShadowViewProj;
        const auto projectionSinY = device->getScreenSpaceSignY() * device->getUVSpaceSignY();
        Mat4::createOrthographicOffCenter(-x, x, -y, y, shadowInfo->nearValue, shadowInfo->farValue, device->getClipSpaceMinZ(), projectionSinY, &matShadowViewProj);

        matShadowViewProj.multiply(matShadowView);
        float shadowInfos[4] = {shadowInfo->size.x, shadowInfo->size.y, (float)shadowInfo->pcfType, shadowInfo->bias};
        memcpy(_shadowUBO.data() + UBOShadow::MAT_LIGHT_VIEW_PROJ_OFFSET, matShadowViewProj.m, sizeof(matShadowViewProj));
        memcpy(_shadowUBO.data() + UBOShadow::SHADOW_INFO_OFFSET, &shadowInfos, sizeof(shadowInfos));
    } else if (mainLight && shadowInfo->getShadowType() == ShadowType::PLANAR) {
        updateDirLight(shadowInfo, mainLight, _shadowUBO);
    }

    memcpy(_shadowUBO.data() + UBOShadow::SHADOW_COLOR_OFFSET, &shadowInfo->color, sizeof(Vec4));

    // update ubos
    _commandBuffers[0]->updateBuffer(_descriptorSet->getBuffer(UBOGlobal::BINDING), _globalUBO.data(), UBOGlobal::SIZE);
    _commandBuffers[0]->updateBuffer(_descriptorSet->getBuffer(UBOShadow::BINDING), _shadowUBO.data(), UBOShadow::SIZE);
}

void DeferredPipeline::updateUBO(RenderView *view) {
    _descriptorSet->update();

    const auto root = GET_ROOT();
    const auto camera = view->getCamera();
    const auto scene = camera->getScene();

    const Light *mainLight = nullptr;
    if (scene->mainLightID) mainLight = scene->getMainLight();

    const auto ambient = _ambient;
    const auto fog = _fog;
    auto &uboGlobalView = _globalUBO;

    const auto shadingWidth = std::floor(_device->getWidth());
    const auto shadingHeight = std::floor(_device->getHeight());

    // update UBOGlobal
    uboGlobalView[UBOGlobal::TIME_OFFSET] = root->cumulativeTime;
    uboGlobalView[UBOGlobal::TIME_OFFSET + 1] = root->frameTime;
    uboGlobalView[UBOGlobal::TIME_OFFSET + 2] = Application::getInstance()->getTotalFrames();

    uboGlobalView[UBOGlobal::SCREEN_SIZE_OFFSET] = _device->getWidth();
    uboGlobalView[UBOGlobal::SCREEN_SIZE_OFFSET + 1] = _device->getHeight();
    uboGlobalView[UBOGlobal::SCREEN_SIZE_OFFSET + 2] = 1.0f / uboGlobalView[UBOGlobal::SCREEN_SIZE_OFFSET];
    uboGlobalView[UBOGlobal::SCREEN_SIZE_OFFSET + 3] = 1.0f / uboGlobalView[UBOGlobal::SCREEN_SIZE_OFFSET + 1];

    uboGlobalView[UBOGlobal::SCREEN_SCALE_OFFSET] = camera->width / shadingWidth * _shadingScale;
    uboGlobalView[UBOGlobal::SCREEN_SCALE_OFFSET + 1] = camera->height / shadingHeight * _shadingScale;
    uboGlobalView[UBOGlobal::SCREEN_SCALE_OFFSET + 2] = 1.0 / uboGlobalView[UBOGlobal::SCREEN_SCALE_OFFSET];
    uboGlobalView[UBOGlobal::SCREEN_SCALE_OFFSET + 3] = 1.0 / uboGlobalView[UBOGlobal::SCREEN_SCALE_OFFSET + 1];

    uboGlobalView[UBOGlobal::NATIVE_SIZE_OFFSET] = shadingWidth;
    uboGlobalView[UBOGlobal::NATIVE_SIZE_OFFSET + 1] = shadingHeight;
    uboGlobalView[UBOGlobal::NATIVE_SIZE_OFFSET + 2] = 1.0f / uboGlobalView[UBOGlobal::NATIVE_SIZE_OFFSET];
    uboGlobalView[UBOGlobal::NATIVE_SIZE_OFFSET + 3] = 1.0f / uboGlobalView[UBOGlobal::NATIVE_SIZE_OFFSET + 1];

    memcpy(uboGlobalView.data() + UBOGlobal::MAT_VIEW_OFFSET, camera->matView.m, sizeof(cc::Mat4));
    memcpy(uboGlobalView.data() + UBOGlobal::MAT_VIEW_INV_OFFSET, camera->getNode()->worldMatrix.m, sizeof(cc::Mat4));
    memcpy(uboGlobalView.data() + UBOGlobal::MAT_PROJ_OFFSET, camera->matProj.m, sizeof(cc::Mat4));
    memcpy(uboGlobalView.data() + UBOGlobal::MAT_PROJ_INV_OFFSET, camera->matProjInv.m, sizeof(cc::Mat4));
    memcpy(uboGlobalView.data() + UBOGlobal::MAT_VIEW_PROJ_OFFSET, camera->matViewProj.m, sizeof(cc::Mat4));
    memcpy(uboGlobalView.data() + UBOGlobal::MAT_VIEW_PROJ_INV_OFFSET, camera->matViewProjInv.m, sizeof(cc::Mat4));
    TO_VEC3(uboGlobalView, camera->position, UBOGlobal::CAMERA_POS_OFFSET);

    auto projectionSignY = _device->getScreenSpaceSignY();
    if (view->getWindow()->hasOffScreenAttachments) {
        projectionSignY *= _device->getUVSpaceSignY(); // need flipping if drawing on render targets
    }
    uboGlobalView[UBOGlobal::CAMERA_POS_OFFSET + 3] = projectionSignY;

    const auto exposure = camera->exposure;
    uboGlobalView[UBOGlobal::EXPOSURE_OFFSET] = exposure;
    uboGlobalView[UBOGlobal::EXPOSURE_OFFSET + 1] = 1.0f / exposure;
    uboGlobalView[UBOGlobal::EXPOSURE_OFFSET + 2] = _isHDR ? 1.0f : 0.0;
    uboGlobalView[UBOGlobal::EXPOSURE_OFFSET + 3] = _fpScale / exposure;

    if (mainLight) {
        TO_VEC3(uboGlobalView, mainLight->direction, UBOGlobal::MAIN_LIT_DIR_OFFSET);
        TO_VEC3(uboGlobalView, mainLight->color, UBOGlobal::MAIN_LIT_COLOR_OFFSET);
        if (mainLight->useColorTemperature) {
            const auto colorTempRGB = mainLight->colorTemperatureRGB;
            uboGlobalView[UBOGlobal::MAIN_LIT_COLOR_OFFSET] *= colorTempRGB.x;
            uboGlobalView[UBOGlobal::MAIN_LIT_COLOR_OFFSET + 1] *= colorTempRGB.y;
            uboGlobalView[UBOGlobal::MAIN_LIT_COLOR_OFFSET + 2] *= colorTempRGB.z;
        }

        if (_isHDR) {
            uboGlobalView[UBOGlobal::MAIN_LIT_COLOR_OFFSET + 3] = mainLight->luminance * _fpScale;
        } else {
            uboGlobalView[UBOGlobal::MAIN_LIT_COLOR_OFFSET + 3] = mainLight->luminance * exposure;
        }
    } else {
        TO_VEC3(uboGlobalView, Vec3::UNIT_Z, UBOGlobal::MAIN_LIT_DIR_OFFSET);
        TO_VEC4(uboGlobalView, Vec4::ZERO, UBOGlobal::MAIN_LIT_COLOR_OFFSET);
    }

    Vec4 skyColor = ambient->skyColor;
    if (_isHDR) {
        skyColor.w = ambient->skyIllum * _fpScale;
    } else {
        skyColor.w = ambient->skyIllum * exposure;
    }
    TO_VEC4(uboGlobalView, skyColor, UBOGlobal::AMBIENT_SKY_OFFSET);

    uboGlobalView[UBOGlobal::AMBIENT_GROUND_OFFSET] = ambient->groundAlbedo.x;
    uboGlobalView[UBOGlobal::AMBIENT_GROUND_OFFSET + 1] = ambient->groundAlbedo.y;
    uboGlobalView[UBOGlobal::AMBIENT_GROUND_OFFSET + 2] = ambient->groundAlbedo.z;
    const auto envmap = _descriptorSet->getTexture((uint)PipelineGlobalBindings::SAMPLER_ENVIRONMENT);
    if (envmap) uboGlobalView[UBOGlobal::AMBIENT_GROUND_OFFSET + 3] = envmap->getLevelCount();

    if (fog->enabled) {
        TO_VEC4(uboGlobalView, fog->fogColor, UBOGlobal::GLOBAL_FOG_COLOR_OFFSET);

        uboGlobalView[UBOGlobal::GLOBAL_FOG_BASE_OFFSET] = fog->fogStart;
        uboGlobalView[UBOGlobal::GLOBAL_FOG_BASE_OFFSET + 1] = fog->fogEnd;
        uboGlobalView[UBOGlobal::GLOBAL_FOG_BASE_OFFSET + 2] = fog->fogDensity;

        uboGlobalView[UBOGlobal::GLOBAL_FOG_ADD_OFFSET] = fog->fogTop;
        uboGlobalView[UBOGlobal::GLOBAL_FOG_ADD_OFFSET + 1] = fog->fogRange;
        uboGlobalView[UBOGlobal::GLOBAL_FOG_ADD_OFFSET + 2] = fog->fogAtten;
    }
}

bool DeferredPipeline::createQuadInputAssembler() {
    // step 1 create vertex buffer
    uint vbStride = sizeof(float) * 4;
    uint vbSize = vbStride * 4;
    _quadVB = _device->createBuffer({gfx::BufferUsageBit::VERTEX | gfx::BufferUsageBit::TRANSFER_DST,
              gfx::MemoryUsageBit::HOST | gfx::MemoryUsageBit::DEVICE, vbSize, vbStride});
    if (!_quadVB) {
        return false;
    }

    float vbData[] = {
        -1.0, -1.0,     // color 0
         0.0,  0.0,     // texCoord 0
         1.0, -1.0,     // color 1
         1.0,  0.0,     // texCoord 1
        -1.0,  1.0,     // color 2
         0.0,  1.0,     // texCoord 2
         1.0,  1.0,     // color 3
         1.0,  1.0,     // texCoord 3
    };
    _quadVB->update(vbData, 0 ,sizeof(vbData));

    // step 2 create index buffer
    uint ibStride = 1;
    uint ibSize = ibStride * 6;

    _quadIB =_device->createBuffer({gfx::BufferUsageBit::INDEX | gfx::BufferUsageBit::TRANSFER_DST,
             gfx::MemoryUsageBit::HOST | gfx::MemoryUsageBit::DEVICE, ibSize, ibStride});

    if (!_quadIB) {
        return false;
    }

    unsigned char ibData[] = {0, 1, 2, 1, 3, 2};
    _quadIB->update(ibData, 0, sizeof(ibData));

    // step 3 create input assembler
    gfx::InputAssemblerInfo info;
    info.attributes.push_back({"a_position", gfx::Format::RG32F});
    info.attributes.push_back({"a_texCoord", gfx::Format::RG32F});
    info.vertexBuffers.push_back(_quadVB);
    info.vertexBuffers.push_back(_quadIB);
    _quadIA = _device->createInputAssembler(info);
    if (!_quadIA) {
        return false;
    }

    return true;
}

void DeferredPipeline::destroyQuadInputAssembler() {
    if (_quadVB) {
        _quadVB->destroy();
        _quadVB = nullptr;
    }

    if (_quadIB) {
        _quadIB->destroy();
        _quadIB = nullptr;
    }

    if (_quadIA) {
        _quadIA->destroy();
        _quadIA = nullptr;
    }
}

bool DeferredPipeline::activeRenderer() {
    _commandBuffers.push_back(_device->getCommandBuffer());

    _globalUBO.fill(0.f);
    _shadowUBO.fill(0.f);

    auto globalUBO = _device->createBuffer({
        gfx::BufferUsageBit::UNIFORM | gfx::BufferUsageBit::TRANSFER_DST,
        gfx::MemoryUsageBit::HOST | gfx::MemoryUsageBit::DEVICE,
        UBOGlobal::SIZE,
        UBOGlobal::SIZE,
        gfx::BufferFlagBit::NONE,
    });
    _descriptorSet->bindBuffer(UBOGlobal::BINDING, globalUBO);

    auto shadowUBO = _device->createBuffer({
        gfx::BufferUsageBit::UNIFORM | gfx::BufferUsageBit::TRANSFER_DST,
        gfx::MemoryUsageBit::HOST | gfx::MemoryUsageBit::DEVICE,
        UBOShadow::SIZE,
        UBOShadow::SIZE,
        gfx::BufferFlagBit::NONE,
    });
    _descriptorSet->bindBuffer(UBOShadow::BINDING, shadowUBO);

    _descriptorSet->update();

    // update global defines when all states initialized.
    _macros.setValue("CC_USE_HDR", _isHDR);
    _macros.setValue("CC_SUPPORT_FLOAT_TEXTURE", _device->hasFeature(gfx::Feature::TEXTURE_FLOAT));

    if (!createQuadInputAssembler()) {
        return false;
    }

    return true;
}

void DeferredPipeline::destroy() {
    destroyQuadInputAssembler();

    if (_descriptorSet) {
        _descriptorSet->getBuffer(UBOGlobal::BINDING)->destroy();
        _descriptorSet->getBuffer(UBOShadow::BINDING)->destroy();
    }

    for (auto &it : _renderPasses) {
        it.second->destroy();
    }
    _renderPasses.clear();

    _commandBuffers.clear();

    CC_SAFE_DELETE(_sphere);

    _shadowFrameBufferMap.clear();

    RenderPipeline::destroy();
}

} // namespace pipeline
} // namespace cc
