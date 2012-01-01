################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../bobyqa/bobyqa.c 

OBJS += \
./bobyqa/bobyqa.o 

C_DEPS += \
./bobyqa/bobyqa.d 


# Each subdirectory must supply rules for building sources it contributes
bobyqa/%.o: ../bobyqa/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


