#pragma once

#include <iostream>
#include <optional>
#include <set>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "timer.hpp"

struct Boilerplate {
  vk::Instance instance;
  vk::PhysicalDevice physicalDevice;
  vk::Device device;
  vk::Queue transferQueue;
  vk::Queue computeQueue;
  uint32_t transferQueueFamily;
  uint32_t computeQueueFamily;

  Boilerplate() {
    ScopedTimer _ = "Initialization bullshit";

    createInstance();
    pickPhysicalDevice();
    createLogicalDevice();
    getQueues();
  }

  ~Boilerplate() {
    device.destroy();
    instance.destroy();
  }

private:
  struct QueueFamilies {
    std::optional<uint32_t> transfer;
    std::optional<uint32_t> compute;
  };

  void createInstance() {
    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = "VulkanHpp Boilerplate";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo = &appInfo;

    instance = vk::createInstance(createInfo);
  }

  void pickPhysicalDevice() {
    auto devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
      throw std::runtime_error("No Vulkan physical devices found!");
    }

    // Just pick the first suitable device
    for (const auto &dev : devices) {
      QueueFamilies qf = findQueueFamilies(dev);
      if (qf.transfer && qf.compute) {
        physicalDevice = dev;
        transferQueueFamily = qf.transfer.value();
        computeQueueFamily = qf.compute.value();
        return;
      }
    }

    throw std::runtime_error(
        "No suitable physical device found with transfer & compute queues");
  }

  QueueFamilies findQueueFamilies(const vk::PhysicalDevice &device) {
    QueueFamilies indices;
    auto queueFamilies = device.getQueueFamilyProperties();

    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilies.size()); i++) {
      const auto &qf = queueFamilies[i];

      if ((qf.queueFlags & vk::QueueFlagBits::eTransfer) &&
          !(qf.queueFlags & vk::QueueFlagBits::eGraphics)) {
        indices.transfer = i;
      }
      if ((qf.queueFlags & vk::QueueFlagBits::eCompute) &&
          !(qf.queueFlags & vk::QueueFlagBits::eGraphics)) {
        indices.compute = i;
      }
    }

    // Fallback if dedicated queues are not available
    if (!indices.transfer) {
      for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eTransfer) {
          indices.transfer = i;
          break;
        }
      }
    }
    if (!indices.compute) {
      for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute) {
          indices.compute = i;
          break;
        }
      }
    }

    return indices;
  }

  void createLogicalDevice() {
    std::set<uint32_t> uniqueQueueFamilies = {transferQueueFamily,
                                              computeQueueFamily};

    float queuePriority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      vk::DeviceQueueCreateInfo queueInfo{};
      queueInfo.queueFamilyIndex = queueFamily;
      queueInfo.queueCount = 1;
      queueInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueInfo);
    }

    vk::PhysicalDeviceFeatures deviceFeatures{};
    vk::DeviceCreateInfo createInfo{};
    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;

    device = physicalDevice.createDevice(createInfo);
  }

  void getQueues() {
    transferQueue = device.getQueue(transferQueueFamily, 0);
    computeQueue = device.getQueue(computeQueueFamily, 0);
  }
};
