################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../api/deprecated.c \
../api/f77api.c \
../api/general.c \
../api/optimize.c \
../api/options.c 

OBJS += \
./api/deprecated.o \
./api/f77api.o \
./api/general.o \
./api/optimize.o \
./api/options.o 

C_DEPS += \
./api/deprecated.d \
./api/f77api.d \
./api/general.d \
./api/optimize.d \
./api/options.d 


# Each subdirectory must supply rules for building sources it contributes
api/%.o: ../api/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


