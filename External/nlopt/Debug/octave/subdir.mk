################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../octave/dummy.c \
../octave/nlopt_optimize-mex.c 

CC_SRCS += \
../octave/nlopt_optimize-oct.cc 

OBJS += \
./octave/dummy.o \
./octave/nlopt_optimize-mex.o \
./octave/nlopt_optimize-oct.o 

C_DEPS += \
./octave/dummy.d \
./octave/nlopt_optimize-mex.d 

CC_DEPS += \
./octave/nlopt_optimize-oct.d 


# Each subdirectory must supply rules for building sources it contributes
octave/%.o: ../octave/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

octave/%.o: ../octave/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


