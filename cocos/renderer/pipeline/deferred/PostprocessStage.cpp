#include "PostprocessStage.h"
#include "DeferredPipeline.h"
#include "../RenderView.h"
#include "gfx/GFXFramebuffer.h"
#include "gfx/GFXCommandBuffer.h"
#include "../helper/SharedMemory.h"
#include "../PipelineStateManager.h"

namespace cc {
namespace pipeline {
RenderStageInfo PostprocessStage::_initInfo = {
    "PostprocessStage",
    static_cast<uint>(DeferredStagePriority::POSTPROCESS),
    0,
};

bool PostprocessStage::initialize(const RenderStageInfo &info) {
    RenderStage::initialize(info);
    return true;
}

void PostprocessStage::activate(RenderPipeline *pipeline, RenderFlow *flow) {
    RenderStage::activate(pipeline, flow);
}

void PostprocessStage::destroy() {
}

void PostprocessStage::render(RenderView *view) {
    DeferredPipeline *pp = dynamic_cast<DeferredPipeline *>(_pipeline);
    assert(pp != nullptr);
    gfx::Device *device = pp->getDevice();
    gfx::CommandBuffer *cmdBf = pp->getCommandBuffers()[0];

    Camera *camera = view->getCamera();
    gfx::Rect renderArea;
    renderArea.x = camera->viewportX * camera->width;
    renderArea.y = camera->viewportY * camera->height;
    renderArea.width = camera->viewportWidth * camera->width * pp->getShadingScale();
    renderArea.height = camera->viewportHeight * camera->height * pp->getShadingScale();
    gfx::Color color = {0, 0, 0, 1};
    gfx::ColorList clst;
    clst.push_back(color);

    gfx::Framebuffer *fb = view->getWindow()->getFramebuffer();
    gfx::RenderPass *rp = fb->getRenderPass() ?
        fb->getRenderPass() : pp->getOrCreateRenderPass(static_cast<gfx::ClearFlags>(camera->clearFlag));

    cmdBf->beginRenderPass(rp, fb, renderArea, clst, camera->clearDepth, camera->clearStencil);
    cmdBf->bindDescriptorSet(static_cast<uint>(SetIndex::GLOBAL), pp->getDescriptorSet());

    // post process
    Root *root = GET_ROOT();
    assert(root != nullptr);
    PassView *pv = GET_PASS(root->deferredPostPass);
    gfx::Shader *sd = GET_SHADER(root->deferredPostPassShader);
    gfx::InputAssembler *ia = pp->getQuadIA();
    gfx::PipelineState *pso = PipelineStateManager::getOrCreatePipelineState(pv, sd, ia, rp);
    assert(pso != nullptr);

    cmdBf->bindPipelineState(pso);
    cmdBf->bindInputAssembler(ia);
    cmdBf->draw(ia);
    cmdBf->endRenderPass();
}
} // namespace pipeline
} // namespace cc