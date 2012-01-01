################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../test/testopt.cpp 

C_SRCS += \
../test/testfuncs.c 

OBJS += \
./test/testfuncs.o \
./test/testopt.o 

C_DEPS += \
./test/testfuncs.d 

CPP_DEPS += \
./test/testopt.d 


# Each subdirectory must supply rules for building sources it contributes
test/%.o: ../test/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

test/%.o: ../test/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


