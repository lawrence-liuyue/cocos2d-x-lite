#include "LightingFlow.h"
#include "DeferredPipeline.h"
#include "LightingStage.h"
#include "SceneCulling.h"

namespace cc {
namespace pipeline {
RenderFlowInfo LightingFlow::_initInfo = {
    "LightingFlow",
    static_cast<uint>(DefferredFlowPriority::LIGHTING),
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

void LightingFlow::activate(RenderPipeline *pipeline) {
    RenderFlow::activate(pipeline);
    if (_lighingRenderPass != nullptr) {
        continue;
    }

    auto device = pipeline->getDevice();
    _width = device->getWidth();
    _height = device->getHeight();

    gfx::RenderPassInfo info;
    info.colorAttachments.push_back({
        gfx::Format::RGBA32F,
        gfx::LoadOp::CLEAR,
        gfx::StoreOp::STORE,
        1,
        gfx::TextureLayout::UNDEFINED,
        gfx::TextureLayout::COLOR_ATTACHMENT_OPTIMAL
    });

    info.depthStencilAttachment = {
        device->getDepthStencilFormat(),
        gfx::LoadOp::LOAD,
        gfx::StoreOp::DISCARD,
        gfx::LoadOp::DISCARD,
        gfx::StoreOp::DISCARD,
        1,
        gfx::TextureLayout::UNDEFINED,
        gfx::TextureLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    };

    _lightingRenderPass = device->createRenderPass(info);
    assert(_lightingRenderPass);

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

    DeferredPipeline *pp = dynamic_cast<DeferredPipeline *>(pipeline);
    _depth = pp->getDepth();

    if (!_lightingFrameBuff) {
        gfx::FramebufferInfo fbInfo;
        fbInfo.renderPass = _lightingRenderPass;
        fbInfo.colorMipmapLevels.push_back(_lightingRenderTarget);
        fbInfo.depthStencilTexture = _depth;
        _lightingFrameBuff = device->createFramebuffer(fbInfo);
        assert(_lightingFrameBuff != nullptr);
    }

    // bind sampler and texture, used in copystage
    pipeline->getDescriptorSet()->bindTexture(PipelineGlobalBindings::SAMPLER_LIGHTING_RESULTMAP,
        _lightingFrameBuffer->getColorTextures()[0]);

    gfx::SamplerInfo spInfo = {
        gfx::Filter::LINEAR,
        gfx::Filter::LINEAR,
        gfx::Filter::NONE,
        gfx::Address::CLAMP,
        gfx::Address::CLAMP,
        gfx::Address::CLAMP
    };

    auto spHash = genSamplerHash(spInfo);
    const gfx::Sampler copySampler = getSampler(spHash);
    pipeline->getDescriptorSet()->bindSampler(PipelineGlobalBindings::SAMPLER_LIGHTING_RESULTMAP, copySampler);
}

void LightingFlow::render(RenderView *view) {
    auto pipeline = static_cast<ForwardPipeline *>(_pipeline);
    pipeline->updateUBOs(view);
    RenderFlow::render(view);
}

void LightingFlow::destroy() {
    RenderFlow::destroy();
}

} // namespace pipeline
} // namespace cc
