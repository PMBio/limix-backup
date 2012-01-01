################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../cdirect/cdirect.c \
../cdirect/hybrid.c 

OBJS += \
./cdirect/cdirect.o \
./cdirect/hybrid.o 

C_DEPS += \
./cdirect/cdirect.d \
./cdirect/hybrid.d 


# Each subdirectory must supply rules for building sources it contributes
cdirect/%.o: ../cdirect/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


