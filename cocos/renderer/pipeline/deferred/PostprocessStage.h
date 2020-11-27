#pragma once

#include "../RenderStage.h"

namespace cc {
namespace pipeline {

class CC_DLL PostprocessStage : public RenderStage {
public:
    PostprocessStage() {};
    ~PostprocessStage() {};

    virtual bool initialize(const RenderStageInfo &info) override;
    virtual void activate(RenderPipeline *pipeline, RenderFlow *flow) override;
    virtual void destroy() override;
    virtual void render(RenderView *view) override;

private:
    gfx::Rect _renderArea;
    static RenderStageInfo _initInfo;
};
} // namespace pipeline
} // namespace cc