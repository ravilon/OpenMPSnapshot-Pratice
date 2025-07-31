#pragma once
#include <glm/glm.hpp>

class VertexPool
{
	static const int GLOBAL_ALIGNMENT = 32;
	static const int LOCAL_ALIGNMENT = 8;

public:
	explicit VertexPool(size_t size);
	~VertexPool();

	template<typename T>
	inline T* alloc(size_t size)
	{
		assert(mCurrentOffset < mPoolSize);
		
		auto alignSize = (size * sizeof(T) - 1 + sizeof(T)) & ~(LOCAL_ALIGNMENT - 1);
		auto address = mPool + mCurrentOffset;
		mCurrentOffset += alignSize;

		return reinterpret_cast<T*>(address);
	};

	void clear();
	template<typename T>
	void clearFromAdr(T* ptr)
	{
		// TODO unsafe - possible missalignment
		mCurrentOffset = static_cast<size_t>(reinterpret_cast<float*>(ptr) - mPool);
	}

private:
	float* mPool;
	size_t mCurrentOffset = 0;
	size_t mPoolSize;
};