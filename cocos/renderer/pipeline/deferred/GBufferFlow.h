#pragma once

#include "../RenderFlow.h"

namespace cc {
namespace pipeline {

class RenderView;
class GBufferStage;
class gfx::RenderPass;

class GBufferFlow : public RenderFlow {
public:
    static const RenderFlowInfo &getInitializeInfo();

    GBufferFlow() = default;
    virtual ~GBufferFlow();

    virtual bool initialize(const RenderFlowInfo &info) override;
    virtual void activate(RenderPipeline *pipeline) override;
    virtual void destroy() override;
    virtual void render(RenderView *view) override;

private:
    void createRenderPass(gfx::device *device);
    void createRenderTargets(gfx::device *device);

private:
    static RenderFlowInfo _initInfo;

    gfx::RenderPass *_gbufferRenderPass = nullptr;
    gfx::TextureList _gbufferRenderTargets;
    gfx::Framebuffer *_gbufferFrameBuffer = nullptr;
    gfx::Texture *_depth = nullptr;
    int _width;
    int _height;

    GBufferStage *_GBufferStage = nullptr;
};

} // namespace pipeline
} // namespace cc
