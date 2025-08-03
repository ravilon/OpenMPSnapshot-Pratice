/*
This file is part of Task-Aware SYCL and is licensed under the terms contained in the COPYING and COPYING.LESSER files.

Copyright (C) 2022-2025 Barcelona Supercomputing Center (BSC)
*/

#include <TASYCL.hpp>

#include "common/QueuePool.hpp"
#include "common/Environment.hpp"
#include "common/TaskingModel.hpp"
#include "common/util/ErrorHandler.hpp"

using namespace tasycl;

#pragma GCC visibility push(default)

void
tasyclCreateQueues(size_t count=TaskingModel::getNumCPUs(), bool shareContext, bool inOrderQueues)
{
sycl::property_list propList = {};
if (inOrderQueues)
propList = {sycl::property::queue::in_order()};

sycl::device dev = sycl::device(sycl::default_selector_v);

if (shareContext)
QueuePool::initialize(count, sycl::context(dev), dev, propList);
else
QueuePool::initialize(count, dev, propList);
}

template<class... Args>
void tasyclCreateQueues(size_t count, Args&&... args)
{
QueuePool::initialize(count, args...);
}

void tasyclCreateQueues(size_t count,
const sycl::property_list& propList)
{
QueuePool::initialize(count, propList);
}

void tasyclCreateQueues(const sycl::async_handler& asyncHandler,
size_t count,
const sycl::property_list& propList)
{
QueuePool::initialize(count, asyncHandler, propList);
}

template <typename DeviceSelector>
void tasyclCreateQueues(const DeviceSelector& deviceSelector,
size_t count,
const sycl::property_list& propList)
{
QueuePool::initialize(count, deviceSelector, propList);
}

template <typename DeviceSelector>
void tasyclCreateQueues(const DeviceSelector& deviceSelector,
const sycl::async_handler& asyncHandler,
size_t count,
const sycl::property_list& propList)
{
QueuePool::initialize(count, deviceSelector, asyncHandler, propList);
}

void tasyclCreateQueues(const sycl::device& syclDevice,
size_t count,
const sycl::property_list& propList)
{
QueuePool::initialize(count, syclDevice, propList);
}

void tasyclCreateQueues(const sycl::device& syclDevice,
const sycl::async_handler& asyncHandler,
size_t count,
const sycl::property_list& propList)
{
QueuePool::initialize(count, syclDevice, asyncHandler, propList);
}

template <typename DeviceSelector>
void tasyclCreateQueues(const sycl::context& syclContext,
const DeviceSelector& deviceSelector,
size_t count,
const sycl::property_list& propList)
{
QueuePool::initialize(count, syclContext, deviceSelector, propList);
}

template <typename DeviceSelector>
void tasyclCreateQueues(const sycl::context& syclContext,
const DeviceSelector& deviceSelector,
const sycl::async_handler& asyncHandler,
size_t count,
const sycl::property_list& propList)
{
QueuePool::initialize(count, syclContext, deviceSelector, asyncHandler, propList);
}

void tasyclCreateQueues(const sycl::context& syclContext,
const sycl::device& syclDevice,
size_t count,
const sycl::property_list& propList)
{
QueuePool::initialize(count, syclContext, syclDevice, propList);
}

void tasyclCreateQueues(const sycl::context& syclContext,
const sycl::device& syclDevice,
const sycl::async_handler& asyncHandler,
size_t count,
const sycl::property_list& propList)
{
QueuePool::initialize(count, syclContext, syclDevice, asyncHandler, propList);
}

void
tasyclDestroyQueues()
{
QueuePool::finalize();
}

void
tasyclGetQueue(sycl::queue *queue, short int queueId /*= TASYCL_QUEUE_ID_DEFAULT*/)
{
assert(queue != nullptr);
*queue = tasyclGetQueue(queueId);
}

sycl::queue&
tasyclGetQueue(short int queueId) {
if(queueId == TASYCL_QUEUE_ID_DEFAULT){
queueId = TaskingModel::getCurrentCPU() % QueuePool::getNumberOfQueues();
}
assert(queueId >= 0);
assert(queueId < QueuePool::getNumberOfQueues());

return QueuePool::getQueue(queueId);
}

void
tasyclReturnQueue(sycl::queue)
{
}

void
tasyclSynchronizeEventAsync(sycl::event e)
{
RequestManager::generateRequest(e, true);
}

void tasyclForeachQueue(std::function<void(sycl::queue)> unary_op)
{
QueuePool::mapQueues(unary_op);
}

#pragma GCC visibility pop
