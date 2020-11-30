#pragma once

#include "../RenderFlow.h"
#include "gfx/GFXRenderPass.h"

namespace cc {
namespace pipeline {

class RenderView;
class GBufferStage;

class GBufferFlow : public RenderFlow {
public:
    static const RenderFlowInfo &getInitializeInfo();

    GBufferFlow() = default;
    virtual ~GBufferFlow();

    virtual bool initialize(const RenderFlowInfo &info) override;
    virtual void activate(RenderPipeline *pipeline) override;
    virtual void destroy() override;
    virtual void render(RenderView *view) override;
    gfx::Framebuffer *getFrameBuffer() {return _gbufferFrameBuffer;}

private:
    void createRenderPass(gfx::Device *device);
    void createRenderTargets(gfx::Device *device);

private:
    static RenderFlowInfo _initInfo;

    gfx::RenderPass *_gbufferRenderPass = nullptr;
    gfx::TextureList _gbufferRenderTargets;
    gfx::Framebuffer *_gbufferFrameBuffer = nullptr;
    gfx::Texture *_depth = nullptr;
    uint _width;
    uint _height;

    GBufferStage *_GBufferStage = nullptr;
};

} // namespace pipeline
} // namespace cc
