#pragma once

#include "../RenderStage.h"

namespace cc {
namespace pipeline {

class RenderFlow;
class RenderView;
class RenderBatchedQueue;
class RenderInstancedQueue;
class RenderAdditiveLightQueue;
class PlanarShadowQueue;

class CC_DLL LightingStage : public RenderStage {
public:
    static const RenderStageInfo &getInitializeInfo();

    LightingStage();
    ~LightingStage();

    virtual bool initialize(const RenderStageInfo &info) override;
    virtual void activate(RenderPipeline *pipeline, RenderFlow *flow) override;
    virtual void destroy() override;
    virtual void render(RenderView *view) override;

    void initLightingBuffer();

private:
    void gatherLights(RenderView *view);

private:
    static RenderStageInfo _initInfo;
    PlanarShadowQueue *_planarShadowQueue = nullptr;
    gfx::Rect _renderArea;
    uint _phaseID = 0;
    uint _transparentPhaseID = 0;

    gfx::Buffer *_deferredLitsBufs = nullptr;
    gfx::Buffer *_deferredLitsBufView = nullptr;
    std::vector<float> _lightBufferData;
    uint _lightBufferStride = 0;
    uint _lightBufferElementCount = 0;
    float _lightMeterScale = 1000.0;
    gfx::DescriptorSet *_descriptorSet = nullptr;
    gfx::DescriptorSetLayout *_descLayout = nullptr;
    uint _maxDeferredLights = UBODeferredLight::LIGHTS_PER_PASS;
};

} // namespace pipeline
} // namespace cc
