#!/bin/bash
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2013, The University of Texas
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name of The University of Texas nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

#
# Makefile
#
# Field G. Van Zee
# 
# Top-level makefile for libflame linear algebra library.
#
#

#
# --- Makefile PHONY target definitions ----------------------------------------
#

.PHONY: all lib install clean \
        check check-config check-fragments check-make-defs \
        install-libs install-headers \
        install-lib-symlinks install-header-symlinks \
        showconfig \
        cleanmost distclean cleanmk cleanleaves \
        changelog



#
# --- Makefile initialization --------------------------------------------------
#

# Define the name of the configuration file.
CONFIG_MK_FILE    := config.mk

# Define the name of the file containing build and architecture-specific
# makefile definitions.
MAKE_DEFS_FILE    := make_defs.mk

# All makefile fragments in the tree will have this name.
FRAGMENT_MK       := .fragment.mk

# Locations of important files.
BUILD_DIR         := build
CONFIG_DIR        := config
FRAME_DIR         := frame
OBJ_DIR           := obj
LIB_DIR           := lib

# The name of directories in which make expects to find special source code
# that must be compiled with out optimizations.
NOOPT_DIR         := noopt

# The text to append to (non-verbose) make output when CFLAGS_NOOPT is used
# instead of CFLAGS.
NOOPT_TEXT        := "(NOTE: optimizations disabled)"

# Construct some paths.
FRAME_PATH        := ./$(FRAME_DIR)
OBJ_PATH          := ./$(OBJ_DIR)
LIB_PATH          := ./$(LIB_DIR)

# CHANGELOG file.
CHANGELOG         := CHANGELOG



#
# --- Include makefile configuration file --------------------------------------
#

# Construct the path to the makefile configuration file that was generated by
# the configure script.
CONFIG_MK_PATH    := ./$(CONFIG_MK_FILE)

# Include the configuration file.
-include $(CONFIG_MK_PATH)

# Detect whether we actually got the configuration file. If we didn't, then
# it is likely that the user has not yet generated it (via configure).
ifeq ($(strip $(CONFIG_MK_INCLUDED)),yes)
CONFIG_MK_PRESENT := yes
else
CONFIG_MK_PRESENT := no
endif

# Now we have access to CONFIG_NAME, which tells us which sub-directory of the
# config directory to use as our configuration.
CONFIG_PATH       := ./$(CONFIG_DIR)/$(CONFIG_NAME)

# Construct base paths for the object file tree.
BASE_OBJ_PATH          := ./$(OBJ_DIR)/$(CONFIG_NAME)
BASE_OBJ_CONFIG_PATH   := $(BASE_OBJ_PATH)/$(CONFIG_DIR)
BASE_OBJ_FRAME_PATH    := $(BASE_OBJ_PATH)/$(FRAME_DIR)

# Construct base path for the library.
BASE_LIB_PATH          := ./$(LIB_DIR)/$(CONFIG_NAME)



#
# --- Include makefile definitions file ----------------------------------------
#

# Construct the path to the makefile definitions file residing inside of
# the configuration sub-directory.
MAKE_DEFS_MK_PATH := $(CONFIG_PATH)/$(MAKE_DEFS_FILE)

# Include the makefile definitions file.
-include $(MAKE_DEFS_MK_PATH)

# Detect whether we actually got the make definitios file. If we didn't, then
# it is likely that the configuration is invalid (or incomplete).
ifeq ($(strip $(MAKE_DEFS_MK_INCLUDED)),yes)
MAKE_DEFS_MK_PRESENT := yes
else
MAKE_DEFS_MK_PRESENT := no
endif



#
# --- Main target variable definitions -----------------------------------------
#


# Construct the architecture-version string, which will be used to name the
# library upon installation.
VERSION                := $(shell cat version)
VERS_CONF              := $(VERSION)-$(CONFIG_NAME)

# --- Library names ---

# Note: These names will be modified later to include the configuration and
# version strings.
BLIS_LIB_NAME      := libblis.a
#BLIS_DLL_NAME      := libblis.so

# --- BLIS framework source variable names ---

# These are the makefile variables that source code files will be accumulated
# into by the makefile fragments. Notice that we include separate variables
# for regular and "noopt" source; the latter stands for "no optimization" and
# is needed because some source code needs to be compiled with a special set
# of compiler flags that avoid optimization (for numerical reasons).
MK_FRAME_SRC           :=
MK_FRAME_NOOPT_SRC     :=
MK_CONFIG_SRC          :=
MK_CONFIG_NOOPT_SRC    :=

# These hold object filenames corresponding to above.
MK_FRAME_OBJS          :=
MK_FRAME_NOOPT_OBJS    :=
MK_CONFIG_OBJS         :=
MK_CONFIG_NOOPT_OBJS   :=

# Append the base library path to the library name.
MK_ALL_BLIS_LIB        := $(BASE_LIB_PATH)/$(BLIS_LIB_NAME)

# --- Define install target names for static libraries ---

MK_BLIS_LIB                  := $(MK_ALL_BLIS_LIB)
MK_BLIS_LIB_INST             := $(patsubst $(BASE_LIB_PATH)/%.a, \
                                           $(INSTALL_PREFIX)/lib/%.a, \
                                           $(MK_BLIS_LIB))
MK_BLIS_LIB_INST_W_VERS_CONF := $(patsubst $(BASE_LIB_PATH)/%.a, \
                                           $(INSTALL_PREFIX)/lib/%-$(VERS_CONF).a, \
                                           $(MK_BLIS_LIB))

# --- Determine which libraries to build ---

MK_LIBS                           :=
MK_LIBS_INST                      :=
MK_LIBS_INST_W_VERS_CONF          :=

ifeq ($(BLIS_ENABLE_STATIC_BUILD),yes)
MK_LIBS                           += $(MK_BLIS_LIB)
MK_LIBS_INST                      += $(MK_BLIS_LIB_INST)
MK_LIBS_INST_W_VERS_CONF          += $(MK_BLIS_LIB_INST_W_VERS_CONF)
endif

# Set the include directory names
MK_INCL_DIR_INST                  := $(INSTALL_PREFIX)/include
MK_INCL_DIR_INST_W_VERS_CONF      := $(INSTALL_PREFIX)/include-$(VERS_CONF)




#
# --- Include makefile fragments -----------------------------------------------
#

# Initialize our list of directory paths to makefile fragments with the empty
# list. This variable will accumulate all of the directory paths in which
# makefile fragments reside.
FRAGMENT_DIR_PATHS :=

# This variable is used by the include statements as they recursively include
# one another. For the framework source tree, we initialize it to the current
# directory since '.' is its parent.
PARENT_PATH        := .

# Recursively include all the makefile fragments in the framework itself.
-include $(addsuffix /$(FRAGMENT_MK), $(FRAME_PATH))

# Now set PARENT_PATH to ./config in preparation to include the fragments in
# the configuration sub-directory.
PARENT_PATH        := ./$(CONFIG_DIR)

# Recursively include all the makefile fragments in the configuration
# sub-directory.
-include $(addsuffix /$(FRAGMENT_MK), $(CONFIG_PATH))

# Create a list of the makefile fragments.
MAKEFILE_FRAGMENTS := $(addsuffix /$(FRAGMENT_MK), $(FRAGMENT_DIR_PATHS))

# Detect whether we actually got any makefile fragments. If we didn't, then it
# is likely that the user has not yet generated them (via configure).
ifeq ($(strip $(MAKEFILE_FRAGMENTS)),)
MAKEFILE_FRAGMENTS_PRESENT := no
else
MAKEFILE_FRAGMENTS_PRESENT := yes
endif



#
# --- Compiler include path definitions ----------------------------------------
#

# Expand the fragment paths that contain .h files to attain the set of header
# files present in all fragment paths.
MK_HEADER_FILES := $(foreach frag_path, $(FRAGMENT_DIR_PATHS), \
                                        $(wildcard $(frag_path)/*.h))

# Strip the leading, internal, and trailing whitespace from our list of header
# files. This makes the "make install-headers" much more readable.
MK_HEADER_FILES := $(strip $(MK_HEADER_FILES))

# Expand the fragment paths that contain .h files, and take the first
# expansion. Then, strip the header filename to leave the path to each header
# location. Notice this process even weeds out duplicates! Add the config
# directory manually since it contains FLA_config.h.
MK_HEADER_DIR_PATHS := $(dir $(foreach frag_path, $(FRAGMENT_DIR_PATHS), \
                                       $(firstword $(wildcard $(frag_path)/*.h))))

# Add -I to each header path so we can specify our include search paths to the
# C compiler.
INCLUDE_PATHS   := $(strip $(patsubst %, -I%, $(MK_HEADER_DIR_PATHS)))
CFLAGS          := $(CFLAGS) $(INCLUDE_PATHS)
CFLAGS_NOOPT    := $(CFLAGS_NOOPT) $(INCLUDE_PATHS)



#
# --- Special preprocessor macro definitions -----------------------------------
#

# Define a C preprocessor macro to communicate the current version so that it
# can be embedded into the library and queried later.
VERS_DEF     := -DBLIS_VERSION_STRING=\"$(VERSION)\"
CFLAGS       := $(CFLAGS) $(VERS_DEF)
CFLAGS_NOOPT := $(CFLAGS) $(VERS_DEF)



#
# --- Library object definitions -----------------------------------------------
#

# Convert source file paths to object file paths by replacing the base source
# directories with the base object directories, and also replacing the source
# file suffix (eg: '.c') with '.o'.
MK_BLIS_CONFIG_OBJS       := $(patsubst $(FRAME_PATH)/%.c, $(BASE_OBJ_FRAME_PATH)/%.o, \
                                        $(filter %.c, $(MK_FRAME_SRC)))
MK_BLIS_CONFIG_NOOPT_OBJS := $(patsubst $(FRAME_PATH)/%.c, $(BASE_OBJ_FRAME_PATH)/%.o, \
                                        $(filter %.c, $(MK_FRAME_NOOPT_SRC)))

MK_BLIS_FRAME_OBJS        := $(patsubst $(CONFIG_PATH)/%.c, $(BASE_OBJ_CONFIG_PATH)/%.o, \
                                        $(filter %.c, $(MK_CONFIG_SRC)))
MK_BLIS_FRAME_NOOPT_OBJS  := $(patsubst $(CONFIG_PATH)/%.c, $(BASE_OBJ_CONFIG_PATH)/%.o, \
                                        $(filter %.c, $(MK_CONFIG_NOOPT_SRC)))

# Combine all of the object files into some readily-accessible variables.
MK_ALL_BLIS_OPT_OBJS      := $(MK_BLIS_CONFIG_OBJS) \
                             $(MK_BLIS_FRAME_OBJS)

MK_ALL_BLIS_NOOPT_OBJS    := $(MK_BLIS_CONFIG_NOOPT_OBJS) \
                             $(MK_BLIS_FRAME_NOOPT_OBJS)

MK_ALL_BLIS_OBJS          := $(MK_ALL_BLIS_OPT_OBJS) \
                             $(MK_ALL_BLIS_NOOPT_OBJS)



#
# --- Targets/rules ------------------------------------------------------------
#

# --- Primary targets ---

all: libs

libs: check $(MK_LIBS)

test: check

install: libs install-libs install-headers \
         install-lib-symlinks install-header-symlinks

clean: cleanmost


# --- Environment check rules ---

check: check-make-defs check-fragments check-config

check-config:
ifeq ($(CONFIG_MK_PRESENT),no)
	$(error Cannot proceed: config.mk not detected! Run configure first)
endif

check-fragments: check-config
ifeq ($(MAKEFILE_FRAGMENTS_PRESENT),no)
	$(error Cannot proceed: makefile fragments not detected! Run configure first)
endif

check-make-defs: check-fragments
ifeq ($(MAKE_DEFS_MK_PRESENT),no)
	$(error Cannot proceed: make_defs.mk not detected! Invalid configuration)
endif


# --- General source code / object code rules ---

$(BASE_OBJ_FRAME_PATH)/%.o: $(FRAME_PATH)/%.c $(MK_HEADER_FILES) $(MAKE_DEFS_MK_PATH)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(CC) $(if $(findstring $(NOOPT_DIR),$@),$(CFLAGS_NOOPT),$(CFLAGS)) -c $< -o $@
else
	@echo "Compiling $<" $(if $(findstring $(NOOPT_DIR),$@),$(NOOPT_TEXT),)
	@$(CC) $(if $(findstring $(NOOPT_DIR),$@),$(CFLAGS_NOOPT),$(CFLAGS)) -c $< -o $@
endif

$(BASE_OBJ_CONFIG_PATH)/%.o: $(CONFIG_PATH)/%.c $(MK_HEADER_FILES) $(MAKE_DEFS_MK_PATH)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(CC) $(if $(findstring $(NOOPT_DIR),$@),$(CFLAGS_NOOPT),$(CFLAGS)) -c $< -o $@
else
	@echo "Compiling $<" $(if $(findstring $(NOOPT_DIR),$@),$(NOOPT_TEXT),)
	@$(CC) $(if $(findstring $(NOOPT_DIR),$@),$(CFLAGS_NOOPT),$(CFLAGS)) -c $< -o $@
endif


# --- Static library archiver rules ---

$(MK_ALL_BLIS_LIB): $(MK_ALL_BLIS_OBJS)
ifeq ($(FLA_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@
else
	@echo "Archiving $@"
	@$(AR) $(ARFLAGS) $@ $?
	@$(RANLIB) $@
endif


# --- Install rules ---

install-libs: check $(MK_LIBS_INST_W_VERS_CONF)

install-headers: check $(MK_INCL_DIR_INST_W_VERS_CONF)

$(MK_INCL_DIR_INST_W_VERS_CONF): $(MK_HEADER_FILES) $(CONFIG_MK_PATH)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(INSTALL) -m 0755 -d $(@)
	$(INSTALL) -m 0644 $(MK_HEADER_FILES) $(@)
else
	@$(INSTALL) -m 0755 -d $(@)
	@echo "Installing C header files into $(@)"
	@$(INSTALL) -m 0644 $(MK_HEADER_FILES) $(@)
endif

$(INSTALL_PREFIX)/lib/%-$(VERS_CONF).a: $(BASE_LIB_PATH)/%.a $(CONFIG_MK_PATH)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(INSTALL) -m 0755 -d $(@D)
	$(INSTALL) -m 0644 $< $@
else
	@echo "Installing $(@F) into $(INSTALL_PREFIX)/lib/"
	@$(INSTALL) -m 0755 -d $(@D)
	@$(INSTALL) -m 0644 $< $@
endif


# --- Install-symlinks rules ---

install-lib-symlinks: check $(MK_LIBS_INST)

install-header-symlinks: check $(MK_INCL_DIR_INST)

$(MK_INCL_DIR_INST): $(MK_INCL_DIR_INST_W_VERS_CONF)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(SYMLINK) $(<F) $(@F)
	$(MV) $(@F) $(INSTALL_PREFIX)
else
	@echo "Installing symlink $(@F) into $(INSTALL_PREFIX)/"
	@$(SYMLINK) $(<F) $(@F)
	@$(MV) $(@F) $(INSTALL_PREFIX)
endif

$(INSTALL_PREFIX)/lib/%.a: $(INSTALL_PREFIX)/lib/%-$(VERS_CONF).a
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(SYMLINK) $(<F) $(@F)
	$(MV) $(@F) $(INSTALL_PREFIX)/lib/
else
	@echo "Installing symlink $(@F) into $(INSTALL_PREFIX)/lib/"
	@$(SYMLINK) $(<F) $(@F)
	@$(MV) $(@F) $(INSTALL_PREFIX)/lib/
endif


# --- Query current configuration ---

showconfig: check
	@echo "Current configuration is '$(CONFIG_NAME)', located in '$(CONFIG_PATH)'"


# --- Clean rules ---

cleanmost: check
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(FIND) $(BASE_OBJ_PATH) -name "*.o" | $(XARGS) $(RM_F)
	- $(FIND) $(BASE_LIB_PATH) -name "*.a" | $(XARGS) $(RM_F)
else
	@echo "Removing .o files from $(BASE_OBJ_PATH)."
	@- $(FIND) $(BASE_OBJ_PATH) -name "*.o" | $(XARGS) $(RM_F)
	@echo "Removing .a files from $(BASE_LIB_PATH)."
	@- $(FIND) $(BASE_LIB_PATH) -name "*.a" | $(XARGS) $(RM_F)
endif

distclean: check cleanmk cleanmost
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(RM_F) $(CONFIG_MK_PATH)
	- $(RM_RF) $(OBJ_PATH)
	- $(RM_RF) $(LIB_PATH)
else
	@echo "Removing $(CONFIG_MK_PATH)."
	@- $(RM_F) $(CONFIG_MK_PATH)
	@echo "Removing $(OBJ_PATH)."
	@- $(RM_RF) $(OBJ_PATH)
	@echo "Removing $(LIB_PATH)."
	@- $(RM_RF) $(LIB_PATH)
endif

cleanmk: check
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(FIND) $(CONFIG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	- $(FIND) $(FRAME_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
else
	@echo "Removing makefile fragments from $(CONFIG_PATH)."
	@- $(FIND) $(CONFIG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	@echo "Removing makefile fragments from $(FRAME_PATH)."
	@- $(FIND) $(FRAME_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
endif


# --- CHANGELOG rules ---

changelog: check
	@echo "Updating '$(CHANGELOG)' via '$(GIT_LOG)'."
	@$(GIT_LOG) > $(CHANGELOG) 


