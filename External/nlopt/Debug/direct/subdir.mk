################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../direct/DIRect.c \
../direct/DIRparallel.c \
../direct/DIRserial.c \
../direct/DIRsubrout.c \
../direct/direct_wrap.c \
../direct/tstc.c 

OBJS += \
./direct/DIRect.o \
./direct/DIRparallel.o \
./direct/DIRserial.o \
./direct/DIRsubrout.o \
./direct/direct_wrap.o \
./direct/tstc.o 

C_DEPS += \
./direct/DIRect.d \
./direct/DIRparallel.d \
./direct/DIRserial.d \
./direct/DIRsubrout.d \
./direct/direct_wrap.d \
./direct/tstc.d 


# Each subdirectory must supply rules for building sources it contributes
direct/%.o: ../direct/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I"/Users/stegle/work/projects/GPmix/External/nlopt/util" -I"/Users/stegle/work/projects/GPmix/External/nlopt" -I"/Users/stegle/work/projects/GPmix/External/nlopt/api" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


