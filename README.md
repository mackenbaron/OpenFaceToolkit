# OpenFaceToolkit
Providing Real-Time Face Processing for Windows

## Using the Pre-Built Windows Tools

I have compiled a full standalone build for Windows 10 x86-64. You can download the program and run it standalone using these binaries. (Download link: https://cdn.iridescent.io/index.php/s/bvi9NXepeckDJ2c)

## Usage

To run the program use: 
```
./OpenFaceToolkit <Face Detection Cascade File> <Facial Landmark Extraction Model File> <MxNet Symbol JSON File> <MxNet Params File> <MxNet Mean File>
```

By default, or in the standalone, you can use the command: 

```
./OpenFaceToolkit haarcascade_frontalface_alt.xml 68.model an_resnet-symbol.json an_resnet-0184.params mean_48.nd
```

## Building From Source

### Requirements

* OpenCV (v. 2.4.13) I have my version of OpenCV built with MSVC 1905 in the folder C:/opencv/. If you're not finding the right DLL files/versions, I would try configuring it in this variety
* MxNet (v. 10.0.1) I have the project configured to find the dependency in C:/tools/mxnet(/build/Release) but the DLL file could be anywhere
* OpenBLAS (v. 0.2.19) Located in C:/tools/OpenBLAS in my configuration, make sure that it has the fortran library.

### Building

To run, build the OpenFaceToolkit Project. This project depends on the LIbLinear and Lib3000 FPS project. The FaceAlignment Project is not used as of right now, but can be used to train your own LBFCascador models (**Warning: When training an LBF Cascador Model, there is a lot involved. Don't attempt this lightly, and the code will certainly not work on your computer without tweaking.**)

Assuming the right include and DLL locations which can be edited by going to: 
```
Project Properties -> C/C++ -> General -> Additional Include Directories
```
or
```
Project Properties -> Linker -> Input -> Additional Dependencies
Project Properties -> Linker -> General -> Additional Library Directories
```
respectively, building the project OpenFaceToolkit should just work. The most likely erorr is that there is an issue with the DLL locations or includes, so depending on the details, that's what you'll get. 

