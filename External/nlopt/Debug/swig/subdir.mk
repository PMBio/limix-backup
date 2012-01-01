################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../swig/nlopt-guile.cpp \
../swig/nlopt-python.cpp 

OBJS += \
./swig/nlopt-guile.o \
./swig/nlopt-python.o 

CPP_DEPS += \
./swig/nlopt-guile.d \
./swig/nlopt-python.d 


# Each subdirectory must supply rules for building sources it contributes
swig/%.o: ../swig/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


