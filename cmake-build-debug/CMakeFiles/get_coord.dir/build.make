# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

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
CMAKE_COMMAND = /home/lxw/Downloads/clion-2021.1.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/lxw/Downloads/clion-2021.1.3/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/ros/A666B94D66B91F4D/ros/learning/get_coord

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/ros/A666B94D66B91F4D/ros/learning/get_coord/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/get_coord.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/get_coord.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/get_coord.dir/flags.make

CMakeFiles/get_coord.dir/main.cpp.o: CMakeFiles/get_coord.dir/flags.make
CMakeFiles/get_coord.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/ros/A666B94D66B91F4D/ros/learning/get_coord/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/get_coord.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/get_coord.dir/main.cpp.o -c /media/ros/A666B94D66B91F4D/ros/learning/get_coord/main.cpp

CMakeFiles/get_coord.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/get_coord.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/ros/A666B94D66B91F4D/ros/learning/get_coord/main.cpp > CMakeFiles/get_coord.dir/main.cpp.i

CMakeFiles/get_coord.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/get_coord.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/ros/A666B94D66B91F4D/ros/learning/get_coord/main.cpp -o CMakeFiles/get_coord.dir/main.cpp.s

# Object files for target get_coord
get_coord_OBJECTS = \
"CMakeFiles/get_coord.dir/main.cpp.o"

# External object files for target get_coord
get_coord_EXTERNAL_OBJECTS =

get_coord: CMakeFiles/get_coord.dir/main.cpp.o
get_coord: CMakeFiles/get_coord.dir/build.make
get_coord: /usr/local/lib/libopencv_stitching.so.3.4.9
get_coord: /usr/local/lib/libopencv_superres.so.3.4.9
get_coord: /usr/local/lib/libopencv_videostab.so.3.4.9
get_coord: /usr/local/lib/libopencv_aruco.so.3.4.9
get_coord: /usr/local/lib/libopencv_bgsegm.so.3.4.9
get_coord: /usr/local/lib/libopencv_bioinspired.so.3.4.9
get_coord: /usr/local/lib/libopencv_ccalib.so.3.4.9
get_coord: /usr/local/lib/libopencv_cvv.so.3.4.9
get_coord: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.9
get_coord: /usr/local/lib/libopencv_dpm.so.3.4.9
get_coord: /usr/local/lib/libopencv_face.so.3.4.9
get_coord: /usr/local/lib/libopencv_freetype.so.3.4.9
get_coord: /usr/local/lib/libopencv_fuzzy.so.3.4.9
get_coord: /usr/local/lib/libopencv_hdf.so.3.4.9
get_coord: /usr/local/lib/libopencv_hfs.so.3.4.9
get_coord: /usr/local/lib/libopencv_img_hash.so.3.4.9
get_coord: /usr/local/lib/libopencv_line_descriptor.so.3.4.9
get_coord: /usr/local/lib/libopencv_optflow.so.3.4.9
get_coord: /usr/local/lib/libopencv_reg.so.3.4.9
get_coord: /usr/local/lib/libopencv_rgbd.so.3.4.9
get_coord: /usr/local/lib/libopencv_saliency.so.3.4.9
get_coord: /usr/local/lib/libopencv_sfm.so.3.4.9
get_coord: /usr/local/lib/libopencv_stereo.so.3.4.9
get_coord: /usr/local/lib/libopencv_structured_light.so.3.4.9
get_coord: /usr/local/lib/libopencv_surface_matching.so.3.4.9
get_coord: /usr/local/lib/libopencv_tracking.so.3.4.9
get_coord: /usr/local/lib/libopencv_xfeatures2d.so.3.4.9
get_coord: /usr/local/lib/libopencv_ximgproc.so.3.4.9
get_coord: /usr/local/lib/libopencv_xobjdetect.so.3.4.9
get_coord: /usr/local/lib/libopencv_xphoto.so.3.4.9
get_coord: /usr/local/lib/libopencv_highgui.so.3.4.9
get_coord: /usr/local/lib/libopencv_videoio.so.3.4.9
get_coord: /usr/local/lib/libopencv_shape.so.3.4.9
get_coord: /usr/local/lib/libopencv_viz.so.3.4.9
get_coord: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.9
get_coord: /usr/local/lib/libopencv_video.so.3.4.9
get_coord: /usr/local/lib/libopencv_datasets.so.3.4.9
get_coord: /usr/local/lib/libopencv_plot.so.3.4.9
get_coord: /usr/local/lib/libopencv_text.so.3.4.9
get_coord: /usr/local/lib/libopencv_dnn.so.3.4.9
get_coord: /usr/local/lib/libopencv_ml.so.3.4.9
get_coord: /usr/local/lib/libopencv_imgcodecs.so.3.4.9
get_coord: /usr/local/lib/libopencv_objdetect.so.3.4.9
get_coord: /usr/local/lib/libopencv_calib3d.so.3.4.9
get_coord: /usr/local/lib/libopencv_features2d.so.3.4.9
get_coord: /usr/local/lib/libopencv_flann.so.3.4.9
get_coord: /usr/local/lib/libopencv_photo.so.3.4.9
get_coord: /usr/local/lib/libopencv_imgproc.so.3.4.9
get_coord: /usr/local/lib/libopencv_core.so.3.4.9
get_coord: CMakeFiles/get_coord.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/ros/A666B94D66B91F4D/ros/learning/get_coord/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable get_coord"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/get_coord.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/get_coord.dir/build: get_coord

.PHONY : CMakeFiles/get_coord.dir/build

CMakeFiles/get_coord.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/get_coord.dir/cmake_clean.cmake
.PHONY : CMakeFiles/get_coord.dir/clean

CMakeFiles/get_coord.dir/depend:
	cd /media/ros/A666B94D66B91F4D/ros/learning/get_coord/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/ros/A666B94D66B91F4D/ros/learning/get_coord /media/ros/A666B94D66B91F4D/ros/learning/get_coord /media/ros/A666B94D66B91F4D/ros/learning/get_coord/cmake-build-debug /media/ros/A666B94D66B91F4D/ros/learning/get_coord/cmake-build-debug /media/ros/A666B94D66B91F4D/ros/learning/get_coord/cmake-build-debug/CMakeFiles/get_coord.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/get_coord.dir/depend

