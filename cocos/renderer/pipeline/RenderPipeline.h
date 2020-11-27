#pragma once

#include "../core/CoreStd.h"
#include "Define.h"
#include "helper/DefineMap.h"
#include "helper/SharedMemory.h"

namespace cc {
namespace gfx {
class CommandBuffer;
class DescriptorSet;
class DescriptorSetLayout;
} // namespace gfx
namespace pipeline {
class DefineMap;
class RenderView;
struct Shadows;

struct CC_DLL RenderPipelineInfo {
    uint tag = 0;
    RenderFlowList flows;
};

class CC_DLL RenderPipeline : public Object {
public:
    static RenderPipeline *getInstance();

    RenderPipeline();
    virtual ~RenderPipeline();

    virtual bool activate();
    virtual void destroy();
    virtual bool initialize(const RenderPipelineInfo &info);
    virtual void render(const vector<RenderView *> &views);

    CC_INLINE const RenderFlowList &getFlows() const { return _flows; }
    CC_INLINE uint getTag() const { return _tag; }
    CC_INLINE const map<String, InternalBindingInst> &getGlobalBindings() const { return _globalBindings; }
    CC_INLINE const DefineMap &getMacros() const { return _macros; }
    CC_INLINE void setValue(const String &name, bool value) { _macros.setValue(name, value); }
    CC_INLINE gfx::DescriptorSet *getDescriptorSet() const { return _descriptorSet; }
    CC_INLINE gfx::DescriptorSetLayout *getDescriptorSetLayout() const { return _descriptorSetLayout; }
    CC_INLINE gfx::Device *getDevice() const {return _device;}

    CC_INLINE Shadows *getShadows() const { return _shadows; }
    CC_INLINE void setShadows(uint shadows) { _shadows = GET_SHADOWS(shadows);}

protected:
    static RenderPipeline *_instance;
    void setDescriptorSetLayout();

    gfx::CommandBufferList _commandBuffers;
    RenderFlowList _flows;
    map<String, InternalBindingInst> _globalBindings;
    DefineMap _macros;
    uint _tag = 0;

    gfx::Device *_device = nullptr;
    gfx::DescriptorSetLayout *_descriptorSetLayout = nullptr;
    gfx::DescriptorSet *_descriptorSet = nullptr;

    Shadows *_shadows = nullptr;
};

} // namespace pipeline
} // namespace cc
