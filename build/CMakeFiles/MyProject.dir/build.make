# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.26.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/matthew/Desktop/Hand-Detection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/matthew/Desktop/Hand-Detection/build

# Include any dependencies generated for this target.
include CMakeFiles/MyProject.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/MyProject.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/MyProject.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MyProject.dir/flags.make

CMakeFiles/MyProject.dir/Main.cpp.o: CMakeFiles/MyProject.dir/flags.make
CMakeFiles/MyProject.dir/Main.cpp.o: /Users/matthew/Desktop/Hand-Detection/Main.cpp
CMakeFiles/MyProject.dir/Main.cpp.o: CMakeFiles/MyProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/matthew/Desktop/Hand-Detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MyProject.dir/Main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MyProject.dir/Main.cpp.o -MF CMakeFiles/MyProject.dir/Main.cpp.o.d -o CMakeFiles/MyProject.dir/Main.cpp.o -c /Users/matthew/Desktop/Hand-Detection/Main.cpp

CMakeFiles/MyProject.dir/Main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MyProject.dir/Main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/matthew/Desktop/Hand-Detection/Main.cpp > CMakeFiles/MyProject.dir/Main.cpp.i

CMakeFiles/MyProject.dir/Main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MyProject.dir/Main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/matthew/Desktop/Hand-Detection/Main.cpp -o CMakeFiles/MyProject.dir/Main.cpp.s

# Object files for target MyProject
MyProject_OBJECTS = \
"CMakeFiles/MyProject.dir/Main.cpp.o"

# External object files for target MyProject
MyProject_EXTERNAL_OBJECTS =

MyProject: CMakeFiles/MyProject.dir/Main.cpp.o
MyProject: CMakeFiles/MyProject.dir/build.make
MyProject: /opt/homebrew/lib/libopencv_gapi.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_stitching.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_alphamat.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_aruco.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_barcode.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_bgsegm.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_bioinspired.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_ccalib.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_dnn_objdetect.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_dnn_superres.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_dpm.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_face.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_freetype.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_fuzzy.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_hfs.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_img_hash.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_intensity_transform.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_line_descriptor.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_mcc.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_quality.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_rapid.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_reg.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_rgbd.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_saliency.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_sfm.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_stereo.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_structured_light.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_superres.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_surface_matching.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_tracking.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_videostab.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_viz.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_wechat_qrcode.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_xfeatures2d.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_xobjdetect.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_xphoto.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_shape.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_highgui.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_datasets.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_plot.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_text.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_ml.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_phase_unwrapping.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_optflow.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_ximgproc.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_video.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_videoio.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_imgcodecs.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_objdetect.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_calib3d.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_dnn.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_features2d.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_flann.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_photo.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_imgproc.4.7.0.dylib
MyProject: /opt/homebrew/lib/libopencv_core.4.7.0.dylib
MyProject: CMakeFiles/MyProject.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/matthew/Desktop/Hand-Detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable MyProject"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MyProject.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MyProject.dir/build: MyProject
.PHONY : CMakeFiles/MyProject.dir/build

CMakeFiles/MyProject.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MyProject.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MyProject.dir/clean

CMakeFiles/MyProject.dir/depend:
	cd /Users/matthew/Desktop/Hand-Detection/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/matthew/Desktop/Hand-Detection /Users/matthew/Desktop/Hand-Detection /Users/matthew/Desktop/Hand-Detection/build /Users/matthew/Desktop/Hand-Detection/build /Users/matthew/Desktop/Hand-Detection/build/CMakeFiles/MyProject.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MyProject.dir/depend

