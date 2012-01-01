################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../stogo/tstc.c 

CC_SRCS += \
../stogo/global.cc \
../stogo/linalg.cc \
../stogo/local.cc \
../stogo/stogo.cc \
../stogo/tools.cc \
../stogo/tst.cc 

OBJS += \
./stogo/global.o \
./stogo/linalg.o \
./stogo/local.o \
./stogo/stogo.o \
./stogo/tools.o \
./stogo/tst.o \
./stogo/tstc.o 

C_DEPS += \
./stogo/tstc.d 

CC_DEPS += \
./stogo/global.d \
./stogo/linalg.d \
./stogo/local.d \
./stogo/stogo.d \
./stogo/tools.d \
./stogo/tst.d 


# Each subdirectory must supply rules for building sources it contributes
stogo/%.o: ../stogo/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

stogo/%.o: ../stogo/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


