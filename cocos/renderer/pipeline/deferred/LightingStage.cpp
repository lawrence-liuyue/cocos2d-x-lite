#include "LightingStage.h"
#include "LightingFlow.h"
#include "../BatchedBuffer.h"
#include "../InstancedBuffer.h"
#include "../PlanarShadowQueue.h"
#include "../RenderBatchedQueue.h"
#include "../RenderInstancedQueue.h"
#include "../RenderQueue.h"
#include "../RenderView.h"
#include "../helper/SharedMemory.h"
#include "DeferredPipeline.h"
#include "gfx/GFXCommandBuffer.h"
#include "gfx/GFXDevice.h"
#include "gfx/GFXFramebuffer.h"
#include "gfx/GFXQueue.h"

namespace cc {
namespace pipeline {
namespace {
void SRGBToLinear(gfx::Color &out, const gfx::Color &gamma) {
    out.x = gamma.x * gamma.x;
    out.y = gamma.y * gamma.y;
    out.z = gamma.z * gamma.z;
}

void LinearToSRGB(gfx::Color &out, const gfx::Color &linear) {
    out.x = std::sqrt(linear.x);
    out.y = std::sqrt(linear.y);
    out.z = std::sqrt(linear.z);
}
} // namespace

RenderStageInfo LightingStage::_initInfo = {
    "LightingStage",
    static_cast<uint>(DeferredStagePriority::LIGHTING),
    static_cast<uint>(RenderFlowTag::SCENE),
    {{false, RenderQueueSortMode::FRONT_TO_BACK, {"default"}},
     {true, RenderQueueSortMode::BACK_TO_FRONT, {"default", "planarShadow"}}}};

const RenderStageInfo &LightingStage::getInitializeInfo() { return LightingStage::_initInfo; }

LightingStage::LightingStage() : RenderStage() {
}

LightingStage::~LightingStage() {
}

bool LightingStage::initialize(const RenderStageInfo &info) {
    RenderStage::initialize(info);
    _renderQueueDescriptors = info.renderQueues;
    _phaseID = getPhaseID("deferred-lighting");
    _transparentPhaseID = getPhaseID("deferred-transparent");
    return true;
}

void LightingStage::gatherLights(RenderView *view) {
    DeferredPipeline *pipeline = dynamic_cast<DeferredPipeline *>(_pipeline);
    if (!pipeline) {
        return;
    }

    gfx::CommandBuffer *cmdBuf = pipeline->getCommandBuffers()[0];
    auto scene = view->getCamera()->getScene();
    const auto sphereLightArrayID = scene->getSphereLightArrayID();
    auto sphereCount = sphereLightArrayID ? sphereLightArrayID[0] : 0;
    const auto spotLightArrayID = scene->getSpotLightArrayID();
    auto spotCount = spotLightArrayID ? spotLightArrayID[0] : 0;

    Sphere sphere;
    auto exposure = view->getCamera()->exposure;
    int idx = 0;
    int fieldLen = 4;
    int totalFieldLen = fieldLen * _maxDeferredLights;
    uint offset = 0;
    cc::Vec4 tmpArray;

    for (int i = 1; i <= sphereCount; i++, idx++) {
        const auto light = scene->getSphereLight(sphereLightArrayID[i]);
        sphere.setCenter(light->position);
        sphere.setRadius(light->range);
        if (!sphere_frustum(&sphere, camera->getFrustum())) {
            continue;
        }
        // position
        offset = idx * fieldLen;
        _lightBufferData[offset] = light->position.x;
        _lightBufferData[offset + 1] = light->position.y;
        _lightBufferData[offset + 2] = light->position.z;
        _lightBufferData[offset + 3] = 0;

        // color
        offset = idx * fieldLenth + totalFieldLen;
        tmpArray.set(light->color.x, light->color.y, light->color.z, 0);
        if (light->useColorTemperature) {
            tmpArray[0] *= light->colorTemperatureRGB.x;
            tmpArray[1] *= light->colorTemperatureRGB.y;
            tmpArray[2] *= light->colorTemperatureRGB.z;
        }

        if (pipeline->isHDR() {
            tmpArray[3] = light->luminance * pipeline->getFpScale() * _lightMeterScale;
        } else {
            tmpArray[3] = light->luminance * exposure * _lightMeterScale;
        }

        for (int i = 0; i < 4; ++i) {
            _lightBufferData[offset + i] = tmpArray[i];
        }

        // size reange angle
        offset = idx * fieldLenth + totalFieldLen * 2;
        _lightBufferData[offset] = light->size;
        _lightBufferData[offset + 1] = light->range;
        _lightBufferData[offset + 2] = 0;
    }

    for (int i = 1; i <= spotCount; i++, idx++) {
        const auto light = scene->getSpotLight(SpotLightArrayID[i]);
        sphere.setCenter(light->position);
        sphere.setRadius(light->range);
        if (!sphere_frustum(&sphere, camera->getFrustum())) {
            continue;
        }
        // position
        offset = idx * fieldLen;
        _lightBufferData[offset] = light->position.x;
        _lightBufferData[offset + 1] = light->position.y;
        _lightBufferData[offset + 2] = light->position.z;
        _lightBufferData[offset + 3] = 1;

        // color
        offset = idx * fieldLenth + totalFieldLen;
        tmpArray.set(light->color.x, light->color.y, light->color.z, 0);
        if (light->useColorTemperature) {
            tmpArray[0] *= light->colorTemperatureRGB.x;
            tmpArray[1] *= light->colorTemperatureRGB.y;
            tmpArray[2] *= light->colorTemperatureRGB.z;
        }

        if (pipeline->isHDR() {
            tmpArray[3] = light->luminance * pipeline->getFpScale() * _lightMeterScale;
        } else {
            tmpArray[3] = light->luminance * exposure * _lightMeterScale;
        }

        for (int i = 0; i < 4; ++i) {
            _lightBufferData[offset + i] = tmpArray[i];
        }

        // size reange angle
        offset = idx * fieldLenth + totalFieldLen * 2;
        _lightBufferData[offset] = light->size;
        _lightBufferData[offset + 1] = light->range;
        _lightBufferData[offset + 2] = light.spotAngle;

        // dir
        offset = idx * fieldLenth + totalFieldLen * 3;
        for (int i = 0; i < 3; ++i) {
            _lightBufferData[offset + i] = light->direction[i];
        }
    }

    _lightBufferData[totalFieldLen * 4] = sphereCount + spotCount;
    cmdBuf->updateBuffer(_deferredLitsBufs, _lightBufferData.data());
}

void LightingStage::activate(RenderPipeline *pipeline, RenderFlow *flow) {
    RenderStage::activate(pipeline, flow);

    auto device = pipeline->getDevice();
    // color/pos/dir/angle 都是vec4存储
    uint totalSize = sizeof(Vec4) * 4 * _maxDeferredLights + sizeof(Vec4);
    totalSize = std::ceil((float)totalSize / device->getUboOffsetAlignment()) * device->getUboOffsetAlignment();

    // create lighting buffer and view
    gfx::BufferInfo bfInfo = {
        gfx::BufferUsageBit::UNIFORM | gfx::BufferUsageBit::TRANSFER_DST,
        gfx::MemoryUsageBit::HOST | gfx::MemoryUsageBit::DEVICE,
        totalSize,
        device->getUboOffsetAlignment(),
    };
    _deferredLitsBufs = device->createBuffer(bfInfo);
    assert(_deferredLitsBufs != nullptr);

    gfx::BufferViewInfo bvInfo = {_deferredLitsBufs, 0, totalSize};
    _deferredLitsBufView = device->createBuffer(bvInfo);
    assert(_deferredLitsBufView != nullptr);

    _deferredLitsBufView->resize(totalSize / sizeof(float));

    // create descriptorset/layout
    gfx::DescriptorSetLayoutInfo layoutInfo = {localDescriptorSetLayout.bindins};
    _descLayout = device->createDescriptorSetLayout(layoutInfo);

    gfx::DescriptorSetInfo setInfo = {_descLayout};
    _descriptorSet = device->createDescriptorSet(setInfo);

    for (const auto &descriptor : _renderQueueDescriptors) {
        uint phase = 0;
        for (const auto &stage : descriptor.stages) {
            phase |= getPhaseID(stage);
        }

        std::function<int(const RenderPass &, const RenderPass &)> sortFunc = opaqueCompareFn;
        switch (descriptor.sortMode) {
            case RenderQueueSortMode::BACK_TO_FRONT:
                sortFunc = transparentCompareFn;
                break;
            case RenderQueueSortMode::FRONT_TO_BACK:
                sortFunc = opaqueCompareFn;
            default:
                break;
        }

        RenderQueueCreateInfo info = {descriptor.isTransparent, phase, sortFunc};
        _renderQueues.emplace_back(CC_NEW(RenderQueue(std::move(info))));
    }

    _planarShadowQueue = CC_NEW(PlanarShadowQueue(_pipeline));
}

void LightingStage::destroy() {
    CC_SAFE_DELETE(_planarShadowQueue);
    RenderStage::destroy();
}

void LightingStage::render(RenderView *view) {
    auto pipeline = static_cast<DeferredPipeline *>(_pipeline);
    auto cmdBuff = pipeline->getCommandBuffers()[0];

    // lighting info
    gatherLights();
    _descriptorSet->bindBuffer(ModelLocalBindings::UBO_DEFERRED_LIGHTS, _deferredLitsBufView);
    _descriptorSet->update();
    cmdBuff->bindDescriptorSet(SetIndex::LOCAL, _descriptorSet);

    // draw quad
    auto camera = view->getCamera();
    gfx::Rect renderArea;
    renderArea.x = camera->viewportX * camera->width;
    renderArea.y = camera->viewportY * camera->height;
    renderArea.width = camera->viewportWidth * camera->width * pipeline->getShadingScale();
    renderArea.height = camera->viewportHeight * camera->Height * pipeline->getShadingScale();

    gfx::Color clearColor = {0.0, 0.0, 0.0, 1.0};
    if (camera->clearFlag | gfx::ClearFlagBit::COLOR) {
        if (pipeline->isHDR()) {
            SRGBToLinear(clearColor, camera->clearColor);
            const auto scale = pipeline->getFpScale() / camera->exposure;
            clearColor.x *= scale;
            clearColor.y *= scale;
            clearColor.z *= scale;
        } else {
            clearColor = camera->clearColor;
        }
    }

    clearColor.w = camera->clearColor.w;

    LightingFlow *flow = dynamic_cast<LightingFlow *>(_flow);
    assert(flow != nullptr);

    auto frameBuffer = flow->getLightingFrameBuffer();
    auto renderPass = frameBuffer->getRenderPass();

    cmdBuff->beginRenderPass(renderPass, frameBuffer, renderArea, &clearColor,
       camera->clearDepth, camera->clearStencil);

    cmdBuff->bindDescriptorSet(SetIndex::GLOBAL, pipeline->getDescriptorSet());

    // get pso and draw quad
    auto pass = GET_PASS(_phaseID);
    auto shader = GET_SHADER(_phaseID);
    gfx::InputAssembler* inputAssembler = pipeline->getQuadIA();

    gfx::PipelineState *pState =PipelineStateManager::getOrCreatePipelineState(
        pass, shader, inputAssembler, renderPass);
    assert(pState != nullptr);

    cmdBuff->bindPipelineState(pState);
    cmdBuff->bindInputAssembler(inputAssembler);
    cmdBuff->draw(inputAssermbler);

    // planerQueue
    _planarShadowQueue->recordCommandBuffer(_device, renderPass, cmdBuff);

    // transparent
    for (auto queue : _renderQueues) {
        queue->clear();
    }
    const auto &renderObjects = pipeline->getRenderObjects();

    uint m = 0, p = 0;
    size_t k = 0;
    for (size_t i = 0; i < renderObjects.size(); ++i) {
        const auto &ro = renderObjects[i];
        const auto model = ro.model;
        const auto subModelID = model->getSubModelID();
        const auto subModelCount = subModelID[0];
        for (m = 1; m <= subModelCount; ++m) {
            auto subModel = model->getSubModelView(subModelID[m]);
            for (p = 0; p < subModel->passCount; ++p) {
                const PassView *pass = subModel->getPassView(p);

                if (pass->phase != _transparentPhaseID) continue;
                for (k i = 0; k < _renderQueues.size(); k++) {
                    RenderQueue[i]->insertRenderPass(ro, m, p);
                }
            }
        }
    }
    for (auto queue : _renderQueues) {
        queue->sort();
        queue->recordCommandBuffer(_device, renderPass, cmdBuff);
    }

    cmdBuff->endRenderPass();
}

} // namespace pipeline
} // namespace cc
