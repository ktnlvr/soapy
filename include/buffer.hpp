#pragma once

#include "boilerplate.hpp"

struct Buffer {
  vk::Buffer buffer;
  vk::DeviceMemory memory;
  vk::DeviceSize size;
};

auto create_buffer(Boilerplate &bp, vk::DeviceSize size) -> Buffer {
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
    if ((memReq.memoryTypeBits & (1 << i)) &&
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

  return buf;
}

void dispatch_compute(Boilerplate &bp, const std::vector<uint32_t> &spirv,
                     const Buffer &outputBuffer, uint32_t x = 1, uint32_t y = 1,
                     uint32_t z = 1) {
  // Create shader module
  vk::ShaderModuleCreateInfo moduleInfo{};
  moduleInfo.codeSize = spirv.size() * sizeof(uint32_t);
  moduleInfo.pCode = spirv.data();
  vk::ShaderModule shaderModule = bp.device.createShaderModule(moduleInfo);

  // Descriptor set layout for storage buffer
  vk::DescriptorSetLayoutBinding layoutBinding{};
  layoutBinding.binding = 0;
  layoutBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
  layoutBinding.descriptorCount = 1;
  layoutBinding.stageFlags = vk::ShaderStageFlagBits::eCompute;

  vk::DescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &layoutBinding;

  vk::DescriptorSetLayout descriptorSetLayout =
      bp.device.createDescriptorSetLayout(layoutInfo);

  // Pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

  vk::PipelineLayout pipelineLayout =
      bp.device.createPipelineLayout(pipelineLayoutInfo);

  // Compute pipeline
  vk::ComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.stage = vk::PipelineShaderStageCreateInfo{
      {}, vk::ShaderStageFlagBits::eCompute, shaderModule, "main"};
  pipelineInfo.layout = pipelineLayout;

  vk::Pipeline pipeline =
      bp.device.createComputePipeline({}, pipelineInfo).value;

  // Descriptor pool
  vk::DescriptorPoolSize poolSize{};
  poolSize.type = vk::DescriptorType::eStorageBuffer;
  poolSize.descriptorCount = 1;

  vk::DescriptorPoolCreateInfo poolInfo{};
  poolInfo.maxSets = 1;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;

  vk::DescriptorPool descriptorPool = bp.device.createDescriptorPool(poolInfo);

  vk::DescriptorSetAllocateInfo allocInfoDS{};
  allocInfoDS.descriptorPool = descriptorPool;
  allocInfoDS.descriptorSetCount = 1;
  allocInfoDS.pSetLayouts = &descriptorSetLayout;

  vk::DescriptorSet descriptorSet =
      bp.device.allocateDescriptorSets(allocInfoDS)[0];

  // Bind buffer
  vk::DescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = outputBuffer.buffer;
  bufferInfo.offset = 0;
  bufferInfo.range = outputBuffer.size;

  vk::WriteDescriptorSet write{};
  write.dstSet = descriptorSet;
  write.dstBinding = 0;
  write.dstArrayElement = 0;
  write.descriptorType = vk::DescriptorType::eStorageBuffer;
  write.descriptorCount = 1;
  write.pBufferInfo = &bufferInfo;

  bp.device.updateDescriptorSets(write, {});

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

  bp.computeQueue.submit(submitInfo, {});
  bp.computeQueue.waitIdle();

  // Cleanup temporary resources
  bp.device.destroyCommandPool(commandPool);
  bp.device.destroyDescriptorPool(descriptorPool);
  bp.device.destroyPipeline(pipeline);
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
