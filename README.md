# kerneldb
This library exposes a C++ class which can be used for understanding CDNA kernel implementations. kernelDB is specifically
implemented to support memory access efficiency analysis as part of the feature-set of omniprobe, from AMD research.

Omniprobe provides intra-kernel observation by injecting code at compile time which causes the instrumented kernel to emit "messages" 
to host code. The instrumented code relies on a buffered I/O capability provided by the dh_comms library implemented as an adjacent project
to Omniprobe. 

Memory access inefficiencies are a common source of performance bottlenecks in GPU kernels. Omniprobe can inject instrumentation which
will cause the kernel to emit memory traces which can be analyzed for such memory access inefficiencies. But Omniprobe instrumentation occurs at the IR
level in LLVM, and many optimizations may occur downstream from where the Omniprobe instrumentation occurs. This implicates how performance
analysis is done. For example, code optimizations for loads routinely gang together individual loads into dwordx4 sized loads. Such optimizations
implicate trace analysis semantics. So understanding how the loads/stores were optimized is a critical aspect of interpreting the memory traces being 
emitted by the instrumented kernel.

Omniprobe messages include a source line number which can be used to identify precisely where various instrumented phenomena are occuring
in the kernel source. kernelDB provides a service for Omniprobe message handlers (i.e. host codes that consume and analyze message streams
from instrumented kernels), which allows message handlers to dereference from the source line in a message to the fully optimized load/store ISA.

kernelDB works by loading hsaco and/or HIP fat binary files and disassembling the CDNA kernels therein. Using the DWARF info contained within
the code objects, kernelDB constructs a map for each kernel which connects source lines to load/store instructions. kernelDB exposes an interface by
which callers can submit a source line number for a kernel and receive back a vector of instruction_t structures containing detailed information
about every load/store operation at that line in the code.
