Enable real-time, offline CV on NXP i.MX8MP using TensorFlow Lite + VX Delegate (NPU) with NNStreamer and FastAPI a modular, hardware-accelerated toolkit for edge vision.
This container enables real-time, offline computer-vision workflows on NXP i.MX 8M Plus. It bundles FastAPI endpoints with a GStreamer/NNStreamer video pipeline and TensorFlow Lite accelerated by the i.MX8MP’s on-chip NPU via the VX Delegate (libvx_delegate.so). Build and deploy detection use cases with reproducible pipelines, low latency, and efficient perf/Watt on the edge no cloud required.

# Object Detection on NXP i.MX

### About Advantech Container Catalog

The Advantech Container Catalog offers plug-and-play, NPU-accelerated container images for Edge AI development on NXP i.MX8MP. These containers abstract hardware complexity, enabling developers to build and deploy AI solutions without worrying about drivers, runtime, or delegate compatibility.

### Key benefits of the Container Catalog include:

| Feature / Benefit | Description |
|-------------------------------------|--------------------------------------------------------------------------|
| Accelerated Edge AI Development | Ready-to-use containerized solutions for fast prototyping and deployment |
| Hardware Compatibility Solved | Eliminates embedded hardware and AI software package incompatibility |
| GPU/NPU Access Ready | Supports passthrough for efficient hardware acceleration |
| Model Conversion & Optimization | Built-in AI model quantization and format conversion support |
| Optimized for CV  | Pre-optimized containers for computer vision and large language models |
| Open Ecosystem | 3rd-party developers can integrate new apps to expand the platform. |

## Container Overview

The Object Detection Runtime on NXP i.MX8MP provides a plug-and-play AI environment for edge vision applications, integrating TensorFlow Lite models (SSD/MobileNet) with NNStreamer pipelines, a FastAPI inference service.

Designed for real-time computer vision on the i.MX NPU, this container offers:

- Offline, on-device inference using TFLite with the VX Delegate (libvx_delegate.so) for NPU acceleration (no internet required post-setup)
- FastAPI middleware for modular CV pipelines and remote control of inference services
- Built-in post-processing modules (bounding box decoding, non-max suppression, class labeling)
- Agent-like orchestration to combine detection with tracking, counting, or rule-based decision logic
- Frame-level context handling to support multi-object tracking and event triggers
- Optional Web UI for real-time video preview and results overlay
- REST API endpoints for seamless integration with external systems
- Customizable model parameters (anchors, thresholds, input resolution) via environment variables or config file

## Container Demo

![Demo](./data/gifs/nxp-det-demo.gif)

## Possible Use Cases

### 1. Industrial Automation
- Detect defective products on assembly lines in real time.
- Monitor safety zones to ensure operators stay clear of robotic arms.
- Track movement of tools or machinery parts for predictive maintenance.

### 2. Smart Retail
- Customer behavior analytics (counting foot traffic, dwell time).
- Shelf monitoring for stock-out detection and planogram compliance.
- Automated checkout by detecting products without barcodes.

### 3. Intelligent Transportation
- Real-time vehicle and pedestrian detection at intersections.
- License plate localization for traffic management.
- Monitoring driver behavior and detecting distractions in cabins.

### 4. Robotics and Drones
- Object detection for autonomous navigation in warehouses or factories.
- Detecting obstacles, pallets, or delivery packages.
- Drone-based inspection of infrastructure (solar panels, power lines, bridges).

### 5. Smart City and Surveillance
- Crowd density estimation and anomaly detection in public spaces.
- Vehicle detection for parking management.
- Perimeter surveillance with alerts for unauthorized entry.

### 6. Healthcare and Assistive Systems
- Detecting PPE compliance (masks, gloves, helmets).
- Monitoring patient activity in elderly care environments.
- Assisting visually impaired individuals by identifying nearby objects.

### 7. Agriculture
- Crop and livestock monitoring through aerial imaging.
- Detecting weeds, pests, or diseases in fields.
- Counting harvested produce in real time for yield estimation.

### 8. Edge AI Research & Development
- Benchmarking new models on NXP NPU accelerators.
- Evaluating trade-offs between INT8 and FP32 inference on real workloads.

## Key Features

- FastAPI Middleware: Modular service layer to expose object detection APIs and orchestrate pipelines
- VX Delegate Integration: Hardware-accelerated inference on the i.MX8MP NPU via TensorFlow Lite
- Complete CV Stack: Pre-integrated support for TFLite and NNStreamer
- Industrial Vision Ready: Optimized GStreamer pipelines with camera/RTSP input, preprocessing, and overlay
- Edge AI Focus: Real-time object detection (SSD, MobileNet) with offline, on-device execution
- Performance Optimized: Tuned specifically for the NXP i.MX8MP (Quad Cortex-A53 + NPU) to deliver low-latency inference and efficient power usage

## Host Device Prerequisites

| Item | Specification |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Compatible Hardware | [Advantech EPC-R3720](https://www.advantech.com/en/products/880a61e5-3fed-41f3-bf53-8be2410c0f19/epc-r3720/mod_fde326be-b36e-4044-ba9a-28c4c49a25c6)
| Host OS | Yocto (5.15-kirkstone)  |
| Required Software Packages | Refer to Below |


## Container Environment Overview

### Software Components on Container Image

| Component   | Version | Description                                                                                  |
|-------------|---------|----------------------------------------------------------------------------------------------|
| LiteRT      | 2.9.1   | Provides TFLite Delegate support for GPU and NPU acceleration                            |
| GStreamer   | 1.20.0  | Multimedia framework for building flexible audio/video pipelines                             |
| NNStreamer   | 2.1.1 | NNStreamer is a pipeline-centric multimedia framework enabling flexible audio/video processing integrated with TFLite and other ML backends.                             |
| Python   | 3.10  | Python runtime for building applications                             |


### Container Quick Start Guide

For container quick start, including the docker-compose file and more, please refer
to [README](https://github.com/Advantech-EdgeSync-Containers/Nagarro-Container-Project/blob/main/Object-Detection-on-NXP-iMX/README.md)

### Supported AI Capabilities

#### Vision Models

| Model                               | Format       |
|-------------------------------------|--------------|
| PoseNet (ResNet50)              | TFLite       | 
| MobileNet V1             | TFLite   |
| SSD Mobilenet V2              | TFLite | 
| Facenet       | TFLite       | 

## Supported AI Model Formats

| Runtime | Format  | Compatible Versions | 
|---------|---------|---------------------|
| LiteRT  | .tflite |       2.9.1        | 

## Hardware Acceleration Support

| Accelerator | Support Level | Compatible Libraries |
|-------------|---------------|----------------------|
| NPU         |  INT8 (primary), limited mixed precision, (INT16/FP16 → quantized internally)       | TensorFlow Lite (VX Delegate), NNStreamer  |             
| GPU         |  FP32 / FP16         | TensorFlow Lite (GPU delegate)    |             

