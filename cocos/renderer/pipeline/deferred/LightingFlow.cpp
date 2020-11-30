#include "LightingFlow.h"
#include "DeferredPipeline.h"
#include "LightingStage.h"
#include "../SceneCulling.h"
#include "gfx/GFXDevice.h"
#include "gfx/GFXDescriptorSet.h"
#include "gfx/GFXFramebuffer.h"

namespace cc {
namespace pipeline {
RenderFlowInfo LightingFlow::_initInfo = {
    "LightingFlow",
    static_cast<uint>(DeferredFlowPriority::LIGHTING),
    static_cast<uint>(RenderFlowTag::SCENE),
    {},
};
const RenderFlowInfo &LightingFlow::getInitializeInfo() { return LightingFlow::_initInfo; }

LightingFlow::~LightingFlow() {
}

bool LightingFlow::initialize(const RenderFlowInfo &info) {
    RenderFlow::initialize(info);

    if (_stages.size() == 0) {
        auto stage = CC_NEW(LightingStage);
        stage->initialize(LightingStage::getInitializeInfo());
        _stages.emplace_back(stage);
    }

    return true;
}

void LightingFlow::createRenderPass(gfx::Device *device) {
    if (_lightingRenderPass == nullptr) {
        gfx::ColorAttachment cAttch = {
            gfx::Format::RGBA32F,
            1,
            gfx::LoadOp::CLEAR,
            gfx::StoreOp::STORE,
            gfx::TextureLayout::UNDEFINED,
            gfx::TextureLayout::COLOR_ATTACHMENT_OPTIMAL
        };
        gfx::RenderPassInfo info;
        info.colorAttachments.push_back(cAttch);
        info.depthStencilAttachment = {
            device->getDepthStencilFormat(),
            1,
            gfx::LoadOp::LOAD,
            gfx::StoreOp::DISCARD,
            gfx::LoadOp::DISCARD,
            gfx::StoreOp::DISCARD,
            gfx::TextureLayout::UNDEFINED,
            gfx::TextureLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        };

        _lightingRenderPass = device->createRenderPass(info);
        assert(_lightingRenderPass != nullptr);
    }
}

void LightingFlow::createFrameBuffer(gfx::Device *device) {
    if (_lightingRenderTarget == nullptr) {
        gfx::TextureInfo rtInfo = {
            gfx::TextureType::TEX2D,
            gfx::TextureUsageBit::COLOR_ATTACHMENT | gfx::TextureUsageBit::SAMPLED,
            gfx::Format::RGBA32F,
            _width,
            _height,
        };
        _lightingRenderTarget = device->createTexture(rtInfo);
    }

    if (!_lightingFrameBuff) {
        gfx::FramebufferInfo fbInfo;
        fbInfo.renderPass = _lightingRenderPass;
        fbInfo.colorTextures.push_back(_lightingRenderTarget);
        fbInfo.depthStencilTexture = _depth;
        _lightingFrameBuff = device->createFramebuffer(fbInfo);
        assert(_lightingFrameBuff != nullptr);
    }
}

void LightingFlow::activate(RenderPipeline *pipeline) {
    RenderFlow::activate(pipeline);
    if (_lightingRenderPass != nullptr) {
        return;
    }

    auto device = pipeline->getDevice();
    _width = device->getWidth();
    _height = device->getHeight();

    DeferredPipeline *pp = dynamic_cast<DeferredPipeline *>(pipeline);
    assert(pp != nullptr);
    _depth = pp->getDepth();

    // create renderpass
    createRenderPass(device);

    // create framebuffer
    createFrameBuffer(device);

    // bind sampler and texture, used in postprocess
    pipeline->getDescriptorSet()->bindTexture(
        static_cast<uint>(PipelineGlobalBindings::SAMPLER_LIGHTING_RESULTMAP),
        _lightingFrameBuff->getColorTextures()[0]);

    gfx::SamplerInfo spInfo = {
        gfx::Filter::LINEAR,
        gfx::Filter::LINEAR,
        gfx::Filter::NONE,
        gfx::Address::CLAMP,
        gfx::Address::CLAMP,
        gfx::Address::CLAMP
    };

    auto spHash = genSamplerHash(spInfo);
    gfx::Sampler *copySampler = getSampler(spHash);
    pipeline->getDescriptorSet()->bindSampler(
        static_cast<uint>(PipelineGlobalBindings::SAMPLER_LIGHTING_RESULTMAP),
        copySampler);
}

void LightingFlow::render(RenderView *view) {
    auto pipeline = dynamic_cast<DeferredPipeline *>(_pipeline);
    pipeline->updateUBOs(view);
    RenderFlow::render(view);
}

void LightingFlow::destroy() {
    RenderFlow::destroy();
}

} // namespace pipeline
} // namespace cc
