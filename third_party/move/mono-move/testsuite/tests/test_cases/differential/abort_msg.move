// RUN: publish --print(bytecode,stackless,micro-ops)
module 0x1::test {
    fun do_abort_msg() {
        abort b"boom"
    }
}

// RUN: execute 0x1::test::do_abort_msg
// CHECK: aborted: code 14566554180833181696 (boom)
