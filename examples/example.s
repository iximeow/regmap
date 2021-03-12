.text
.global main
.align 16
main:
_main:
  call map_registers

  # load up the sad message in case this doesn't work...
  leaq old(%rip), %rsi

  xor %ecx, %ecx
  movq $1, 56   # set the write fd to 1 (stdout)
  leaq new(%rip), %rsp
  mov %rsp, 48  # set the message pointer in rsi
  movq $32, 16  # set the message length in rdx
  movq $1, 0    # set `write` as the syscall number
  syscall

  movq $60, 0   # syscall = NR_exit
  movq $0, 56   # err = 0
  syscall

.data
old:
.ascii "it didn't work :(            \n"

new:
.ascii "memory-mapped registers work!\n"


