#pragma once

#include "../RenderFlow.h"

namespace cc {
namespace pipeline {

class RenderView;

class LightingFlow : public RenderFlow {
public:
    static const RenderFlowInfo &getInitializeInfo();

    LightingFlow() = default;
    virtual ~LightingFlow();

    virtual bool initialize(const RenderFlowInfo &info) override;
    virtual void activate(RenderPipeline *pipeline) override;
    virtual void destroy() override;
    virtual void render(RenderView *view) override;

    gfx::Framebuffer *getLightingFrameBuffer(){return _lightingFrameBuff;}

private:
    static RenderFlowInfo _initInfo;

    gfx::RenderPass  *_lightingRenderPass = nullptr;
    gfx::Texture *_lightingRenderTarget = nullptr;
    gfx::Texture *_depth = nullptr;
    gfx::Framebuffer *_lightingFrameBuff = nullptr;
    uint _width;
    uint _height;
};

} // namespace pipeline
} // namespace cc
