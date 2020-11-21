#pragma once

#include "../RenderStage.h"
#include "../../core/gfx/GFXFramebuffer.h"

namespace cc {
namespace pipeline {

class RenderFlow;
class RenderView;
class RenderBatchedQueue;
class RenderInstancedQueue;
class RenderAdditiveLightQueue;
class PlanarShadowQueue;
class ForwardPipeline;

class CC_DLL GBufferStage : public RenderStage {
public:
    static const RenderStageInfo &getInitializeInfo();

    ForwardStage();
    ~ForwardStage();

    virtual bool initialize(const RenderStageInfo &info) override;
    virtual void activate(RenderPipeline *pipeline, RenderFlow *flow) override;
    virtual void destroy() override;
    virtual void render(RenderView *view) override;

    void getGbufferFrameBuffer(gfx::Framebuffer *fb) {_gbufferFrameBuffer = fb;}

private:
    static RenderStageInfo _initInfo;
    PlanarShadowQueue *_planarShadowQueue = nullptr;
    RenderBatchedQueue *_batchedQueue = nullptr;
    RenderInstancedQueue *_instancedQueue = nullptr;
    gfx::Rect _renderArea;
    uint _phaseID = 0;

    gfx::Framebuffer *_gbufferFrameBuffer = nullptr;
};

} // namespace pipeline
} // namespace cc