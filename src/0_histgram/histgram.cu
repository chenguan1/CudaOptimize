
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <assert.h>
#include <iostream>
#include <time.h>

using namespace std;

__global__ void kernel_histgram_01(unsigned char* buffer, int size, unsigned int* histo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int tid = idx + idy * blockDim.x * gridDim.x;
    if (tid >= size) return;

    unsigned char v = buffer[tid];
    atomicAdd(&histo[v], 1);
}

/*
һ���߳�����32�߳�
ÿ����߳����Ķ������ϲ�����һ�Σ����δ�����С32�ֽڣ����128���ֽ�
128 / 16 = 8��ÿ���̴߳���8���ֽ�Ч�����
*/
__global__ void kernel_histgram_02(unsigned char* buffer, int size, unsigned int* histo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int tid = idx + idy * blockDim.x * gridDim.x * 8;
    if (tid >= size) return;

    long long v = *(long long*)(&buffer[tid]);
    atomicAdd(&histo[v & 0x00000000000000FF], 1);
    atomicAdd(&histo[v & 0x000000000000FF00 >> 8], 1);
    atomicAdd(&histo[v & 0x0000000000FF0000 >> 16], 1);
    atomicAdd(&histo[v & 0x00000000FF000000 >> 24], 1);
    atomicAdd(&histo[v & 0x000000FF00000000 >> 32], 1);
    atomicAdd(&histo[v & 0x0000FF0000000000 >> 40], 1);
    atomicAdd(&histo[v & 0x00FF000000000000 >> 48], 1);
    atomicAdd(&histo[v & 0xFF00000000000000 >> 56], 1);
}

/*
ƿ������atomicAdd�����ԭ�Ӳ������������˲��е�������
�����ö�����������ڴ����atomicAdd�����оֲ����ܣ�
�����ջ�ʹ��atomicAdd������atomicAdd�ĵ��ô�����
*/
__shared__ unsigned int d_bin_data_shared[256];
__global__ void kernel_histgram_03(unsigned char* buffer, int size, unsigned int* histo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int tid = idx + idy * blockDim.x * gridDim.x * 8;
    if (tid >= size) return;

    int tidInBlock = threadIdx.y * blockDim.x + threadIdx.x;
    if (tidInBlock < 256) {
        d_bin_data_shared[tidInBlock] = 0;
    }
    __syncthreads();

    unsigned char v = buffer[tid];
    ++d_bin_data_shared[v];
    __syncthreads();

    atomicAdd(&histo[tidInBlock], d_bin_data_shared[tidInBlock]);
}

void histgram_cpu(unsigned char* buffer, int size, unsigned int* histo)
{
    clock_t st = clock();
    cout << "caculate with cpu:" << endl;
    for (int i = 0; i < size; i++) {
        auto v = buffer[i];
        ++histo[v];
    }
    cout << "  timespend " << clock() - st << " ms" << endl;
}

int main()
{
    unsigned char *h_hist_data = nullptr;
    unsigned char *d_hist_data = nullptr;

    unsigned int h_bin_data[256];
    unsigned int *d_bin_data = nullptr;

    // ׼������
    cout << "preparing data:" << endl;
    int length = 1E+8;
    h_hist_data = new unsigned char[length];
    memset(h_hist_data, 0x00, length);

    cudaMalloc(&d_hist_data, length);
    cudaMalloc(&d_bin_data, 256 * sizeof(int));
    cudaMemset(d_bin_data, 0, 255 * sizeof(int));

    for (int i = 0; i < length; i++) {
        //auto v = (unsigned char)(rand() % 250 + 2);
        auto v = (unsigned char)(i % 256);
        h_hist_data[i] = v;
    }
    cudaMemcpy(d_hist_data, h_hist_data, length, cudaMemcpyHostToDevice);
    cout << "  ok" << endl;

    // cpu����
    histgram_cpu(h_hist_data, length, h_bin_data);

    // ��ʱ����
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);


    // ���ü���
    cout << "caculate with gpu:"<< endl;
    /*const int threadCount = 256;
    dim3 tn(threadCount);
    dim3 bn(length / threadCount + 1);
    cudaEventRecord(ev0);
    kernel_histgram_01 << <bn, tn >> >(d_hist_data, length, d_bin_data);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);*/

    // �߳����Ż���Ч������
    /*const int threadCount = 256;
    dim3 tn(threadCount);
    dim3 bn(length / 8 / threadCount + 1);
    cudaEventRecord(ev0);
    kernel_histgram_02 << <bn, tn >> >(d_hist_data, length, d_bin_data);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);*/

    const int threadCount = 512;
    dim3 tn(threadCount);
    dim3 bn(length / threadCount + 1);
    cudaEventRecord(ev0);
    kernel_histgram_03 << <bn, tn >> >(d_hist_data, length, d_bin_data);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);

    // ����
    cout << "============analysis===========" << endl;
    cudaMemcpy(h_bin_data, d_bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 256; i++) {
        cout << i << " : " <<h_bin_data[i] << endl;
    }


    // ��ʱ�׶�
    float timespend = 0;
    cudaEventElapsedTime(&timespend, ev0, ev1);
    cout << "  timespend " << timespend << " ms" << endl;

    cin.ignore();

    return 0;
}