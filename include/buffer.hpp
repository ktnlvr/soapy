#pragma once

#include <unordered_map>
#include <fstream>
#include <span>

#include "boilerplate.hpp"

struct Buffer {
  vk::Buffer buffer;
  vk::DeviceMemory memory;
  vk::DeviceSize size;
};
template<typename T = void>
auto create_buffer(Boilerplate &bp, vk::DeviceSize size, const T* data = nullptr) -> Buffer {
  if constexpr (!std::is_void_v<T>) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "create_buffer: T must be trivially copyable");
  }

  Buffer buf{};
  buf.size = size;

  vk::BufferCreateInfo bufferInfo{};
  bufferInfo.size = size;
  bufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eTransferSrc |
                     vk::BufferUsageFlagBits::eTransferDst;
  bufferInfo.sharingMode = vk::SharingMode::eExclusive;

  buf.buffer = bp.device.createBuffer(bufferInfo);

  vk::MemoryRequirements memReq =
      bp.device.getBufferMemoryRequirements(buf.buffer);

  vk::PhysicalDeviceMemoryProperties memProps =
      bp.physicalDevice.getMemoryProperties();

  uint32_t memTypeIndex = 0;
  bool found = false;
  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
    if ((memReq.memoryTypeBits & (1u << i)) &&
        (memProps.memoryTypes[i].propertyFlags &
         (vk::MemoryPropertyFlagBits::eHostVisible |
          vk::MemoryPropertyFlagBits::eHostCoherent))) {
      memTypeIndex = i;
      found = true;
      break;
    }
  }

  if (!found)
    throw std::runtime_error("No suitable memory type found!");

  vk::MemoryAllocateInfo allocInfo{};
  allocInfo.allocationSize = memReq.size;
  allocInfo.memoryTypeIndex = memTypeIndex;

  buf.memory = bp.device.allocateMemory(allocInfo);
  bp.device.bindBufferMemory(buf.buffer, buf.memory, 0);

  // If user provided a pointer, copy 'size' bytes (clamped to buffer size)
  if (data != nullptr) {
    vk::DeviceSize copyBytes = std::min<vk::DeviceSize>(buf.size, size);
    try {
      void* mapped = bp.device.mapMemory(buf.memory, 0, copyBytes);
      std::memcpy(mapped, reinterpret_cast<const void*>(data), static_cast<size_t>(copyBytes));
      bp.device.unmapMemory(buf.memory);
    } catch (vk::SystemError &e) {
      if (e.code() == vk::Result::eErrorOutOfHostMemory)
        throw std::runtime_error("Out of host memory while mapping buffer!");
      else
        throw;
    }
  }

  return buf;
}


void dispatch_compute(const char* name, Boilerplate &bp, const std::vector<uint32_t> &spirv,
                      const std::vector<Buffer> &storageBuffers,
                      const std::unordered_map<uint32_t, Buffer> &uniformBuffers,
                      uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) {
  // Create shader module
  vk::ShaderModuleCreateInfo moduleInfo{};
  moduleInfo.codeSize = spirv.size() * sizeof(uint32_t);
  moduleInfo.pCode = spirv.data();
  vk::ShaderModule shaderModule = bp.device.createShaderModule(moduleInfo);

  // Build descriptor set layout bindings:
  // - storageBuffers are assigned bindings [0 .. storageBuffers.size()-1]
  // - uniformBuffers use the user-specified binding key
  size_t numStorage = storageBuffers.size();
  size_t numUniforms = uniformBuffers.size();

  if (numStorage == 0 && numUniforms == 0) {
    throw std::runtime_error("dispatch_compute: no buffers or uniforms provided");
  }

  // Validate: ensure no binding collision (uniform using an already-used storage binding)
  for (const auto &ub : uniformBuffers) {
    uint32_t binding = ub.first;
    if (binding < numStorage) {
      throw std::runtime_error("dispatch_compute: uniform binding collides with storage buffer binding");
    }
  }

  std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
  layoutBindings.reserve(numStorage + numUniforms);

  // storage buffer bindings: 0 .. numStorage-1
  for (uint32_t i = 0; i < static_cast<uint32_t>(numStorage); ++i) {
    vk::DescriptorSetLayoutBinding b{};
    b.binding = i;
    b.descriptorType = vk::DescriptorType::eStorageBuffer;
    b.descriptorCount = 1;
    b.stageFlags = vk::ShaderStageFlagBits::eCompute;
    layoutBindings.push_back(b);
  }

  // uniform buffer bindings: at user-specified binding numbers
  for (const auto &p : uniformBuffers) {
    uint32_t binding = p.first;
    vk::DescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = vk::DescriptorType::eUniformBuffer;
    b.descriptorCount = 1;
    b.stageFlags = vk::ShaderStageFlagBits::eCompute;
    layoutBindings.push_back(b);
  }

  // Create descriptor set layout
  vk::DescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
  layoutInfo.pBindings = layoutBindings.data();

  vk::DescriptorSetLayout descriptorSetLayout =
      bp.device.createDescriptorSetLayout(layoutInfo);

  // Pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

  vk::PipelineLayout pipelineLayout =
      bp.device.createPipelineLayout(pipelineLayoutInfo);

  const std::string cacheFilename = "pipeline_cache.tmp";
  std::vector<uint8_t> initialCacheData;
  {
    std::ifstream in(cacheFilename, std::ios::binary | std::ios::ate);
    if (in.good()) {
      std::streamsize size = in.tellg();
      in.seekg(0, std::ios::beg);
      if (size > 0) {
        initialCacheData.resize(static_cast<size_t>(size));
        if (!in.read(reinterpret_cast<char*>(initialCacheData.data()), size)) {
          initialCacheData.clear();
        }
      }
    }
  }

  vk::PipelineCacheCreateInfo cacheCreateInfo{};
  if (!initialCacheData.empty()) {
    cacheCreateInfo.initialDataSize = initialCacheData.size();
    cacheCreateInfo.pInitialData = initialCacheData.data();
  }

  vk::PipelineCache pipelineCache =
      bp.device.createPipelineCache(cacheCreateInfo);

  // Compute pipeline
  vk::ComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.stage = vk::PipelineShaderStageCreateInfo{
      {}, vk::ShaderStageFlagBits::eCompute, shaderModule, "main"};
  pipelineInfo.layout = pipelineLayout;

  vk::Pipeline pipeline =
      bp.device.createComputePipeline(pipelineCache, pipelineInfo).value;

  // After pipeline creation, extract and save pipeline cache back to disk
  try {
    std::vector<uint8_t> cacheData = bp.device.getPipelineCacheData(pipelineCache);
    if (!cacheData.empty()) {
      std::ofstream out(cacheFilename, std::ios::binary | std::ios::trunc);
      if (out.good()) {
        out.write(reinterpret_cast<const char*>(cacheData.data()),
                  static_cast<std::streamsize>(cacheData.size()));
        out.close();
      }
    }
  } catch (const std::exception &) {
    // Some drivers may fail here; ignore to stay robust
  }

  // Descriptor pool: prepare pool sizes for storage and uniform buffers
  std::vector<vk::DescriptorPoolSize> poolSizes;
  if (numStorage > 0) {
    vk::DescriptorPoolSize s{};
    s.type = vk::DescriptorType::eStorageBuffer;
    s.descriptorCount = static_cast<uint32_t>(numStorage);
    poolSizes.push_back(s);
  }
  if (numUniforms > 0) {
    vk::DescriptorPoolSize u{};
    u.type = vk::DescriptorType::eUniformBuffer;
    u.descriptorCount = static_cast<uint32_t>(numUniforms);
    poolSizes.push_back(u);
  }

  vk::DescriptorPoolCreateInfo poolInfo{};
  poolInfo.maxSets = 1;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.empty() ? nullptr : poolSizes.data();

  vk::DescriptorPool descriptorPool = bp.device.createDescriptorPool(poolInfo);

  // Allocate descriptor set
  vk::DescriptorSetAllocateInfo allocInfoDS{};
  allocInfoDS.descriptorPool = descriptorPool;
  allocInfoDS.descriptorSetCount = 1;
  allocInfoDS.pSetLayouts = &descriptorSetLayout;

  vk::DescriptorSet descriptorSet =
      bp.device.allocateDescriptorSets(allocInfoDS)[0];

  // Prepare descriptor buffer infos and write descriptors
  // We collect buffer infos first to ensure pointer stability while creating writes.
  std::vector<vk::DescriptorBufferInfo> bufferInfos;
  bufferInfos.reserve(numStorage + numUniforms);

  std::vector<vk::WriteDescriptorSet> writes;
  writes.reserve(numStorage + numUniforms);

  // storage buffers: bindings 0..N-1
  for (uint32_t i = 0; i < static_cast<uint32_t>(numStorage); ++i) {
    const Buffer &buf = storageBuffers[i];
    vk::DescriptorBufferInfo bi{};
    bi.buffer = buf.buffer;
    bi.offset = 0;
    bi.range = buf.size;
    bufferInfos.push_back(bi);

    vk::WriteDescriptorSet w{};
    w.dstSet = descriptorSet;
    w.dstBinding = i;
    w.dstArrayElement = 0;
    w.descriptorType = vk::DescriptorType::eStorageBuffer;
    w.descriptorCount = 1;
    w.pBufferInfo = &bufferInfos.back();
    writes.push_back(w);
  }

  // uniform buffers: respect user-specified binding keys
  for (const auto &p : uniformBuffers) {
    uint32_t binding = p.first;
    const Buffer &buf = p.second;
    vk::DescriptorBufferInfo bi{};
    bi.buffer = buf.buffer;
    bi.offset = 0;
    bi.range = buf.size;
    bufferInfos.push_back(bi);

    vk::WriteDescriptorSet w{};
    w.dstSet = descriptorSet;
    w.dstBinding = binding;
    w.dstArrayElement = 0;
    w.descriptorType = vk::DescriptorType::eUniformBuffer;
    w.descriptorCount = 1;
    // pBufferInfo points into bufferInfos; safe because bufferInfos won't be modified afterwards
    w.pBufferInfo = &bufferInfos.back();
    writes.push_back(w);
  }

  // Update descriptor sets
  bp.device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  // Command pool + buffer
  vk::CommandPoolCreateInfo cmdPoolInfo{};
  cmdPoolInfo.queueFamilyIndex = bp.computeQueueFamily;
  cmdPoolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  vk::CommandPool commandPool = bp.device.createCommandPool(cmdPoolInfo);

  vk::CommandBufferAllocateInfo cmdBufAlloc{};
  cmdBufAlloc.commandPool = commandPool;
  cmdBufAlloc.level = vk::CommandBufferLevel::ePrimary;
  cmdBufAlloc.commandBufferCount = 1;

  vk::CommandBuffer cmdBuffer =
      bp.device.allocateCommandBuffers(cmdBufAlloc)[0];

  vk::CommandBufferBeginInfo beginInfo{};
  cmdBuffer.begin(beginInfo);

  cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
  cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout,
                               0, descriptorSet, {});
  cmdBuffer.dispatch(x, y, z);

  cmdBuffer.end();

  // Submit and wait
  vk::SubmitInfo submitInfo{};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuffer;

  {
    ScopedTimer _ = name;
    bp.computeQueue.submit(submitInfo, {});
    bp.computeQueue.waitIdle();
  }

  // Cleanup temporary resources
  bp.device.destroyCommandPool(commandPool);
  bp.device.destroyDescriptorPool(descriptorPool);
  bp.device.destroyPipeline(pipeline);
  bp.device.destroyPipelineCache(pipelineCache);
  bp.device.destroyPipelineLayout(pipelineLayout);
  bp.device.destroyDescriptorSetLayout(descriptorSetLayout);
  bp.device.destroyShaderModule(shaderModule);
}


std::vector<float> read_buffer(Boilerplate &bp, const Buffer &buf) {
    try {
        void* data = bp.device.mapMemory(buf.memory, 0, buf.size);
        size_t count = buf.size / sizeof(float);
        std::vector<float> result(count);
        std::memcpy(result.data(), data, buf.size);
        bp.device.unmapMemory(buf.memory);
        return result;
    } catch (vk::SystemError& e) {
        if (e.code() == vk::Result::eErrorOutOfHostMemory)
            throw std::runtime_error("Out of host memory while mapping buffer!");
        else
            throw;
    }
}
