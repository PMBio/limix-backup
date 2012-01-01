################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../luksan/mssubs.c \
../luksan/plip.c \
../luksan/plis.c \
../luksan/pnet.c \
../luksan/pssubs.c 

OBJS += \
./luksan/mssubs.o \
./luksan/plip.o \
./luksan/plis.o \
./luksan/pnet.o \
./luksan/pssubs.o 

C_DEPS += \
./luksan/mssubs.d \
./luksan/plip.d \
./luksan/plis.d \
./luksan/pnet.d \
./luksan/pssubs.d 


# Each subdirectory must supply rules for building sources it contributes
luksan/%.o: ../luksan/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


