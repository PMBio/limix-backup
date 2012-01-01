################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../util/mt19937ar.c \
../util/qsort_r.c \
../util/redblack.c \
../util/redblack_test.c \
../util/rescale.c \
../util/sobolseq.c \
../util/stop.c \
../util/timer.c 

OBJS += \
./util/mt19937ar.o \
./util/qsort_r.o \
./util/redblack.o \
./util/redblack_test.o \
./util/rescale.o \
./util/sobolseq.o \
./util/stop.o \
./util/timer.o 

C_DEPS += \
./util/mt19937ar.d \
./util/qsort_r.d \
./util/redblack.d \
./util/redblack_test.d \
./util/rescale.d \
./util/sobolseq.d \
./util/stop.d \
./util/timer.d 


# Each subdirectory must supply rules for building sources it contributes
util/%.o: ../util/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


