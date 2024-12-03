# kerneldb
This library exposes a C++ class which can be used for querying data within CDNA kernel implementations. kernelDB is initially
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
## Usage
To use kernelDB, simply create an instance of the kernelDB class, providing an hsa_agent_t and a std::string& of the file name containing the kernels you want
load. If the file name is "", kernelDB will look inside the running process and all associated shared libraries for hip fat binary bundles and load any kernels
it finds there. kernelDB can handle both fat binaries as well as stand alone code objects (e.g. hsaco files)

Once a kernelDB instance has been created, data regarding all load/store instructions can be queried by providing the kernel name of interest, and the line number
you're interested in. 

Under the test directory there is a small test program (kdbtest) that exercises the API and can serve as an example of one way to use the api. 
```
#include <iostream>
#include <string>
#include "inc/kernelDB.h"

int main(int argc, char **argv)
{
    hsa_init();
    if (argc > 1)
    {
        std::string str(argv[1]);
        hsa_agent_t agent;
        if(hsa_iterate_agents ([](hsa_agent_t agent, void *data){
                    hsa_agent_t *this_agent  = reinterpret_cast<hsa_agent_t *>(data);
                    *this_agent = agent;
                    return HSA_STATUS_SUCCESS;
                }, reinterpret_cast<void *>(&agent))== HSA_STATUS_SUCCESS)
        {
            kernelDB::kernelDB test(agent,str);
            std::vector<std::string> kernels;
            std::vector<uint32_t> lines;
            test.getKernels(kernels);
            for (auto kernel : kernels)
            {
                std::vector<uint32_t> lines;
                test.getKernelLines(kernel, lines);
                for (auto& line : lines)
                {
                    std::cout << "Line for " << kernel << " " << line << std::endl;
                    auto inst = test.getInstructionsForLine(kernel, line);
                    for (auto item : inst)
                    {
                        std::cout << "Disassembly: " << item.disassembly_ << std::endl;
                    }
                }
            }
        }
    }
    else
        std::cout << "Usage: kdbtest <hsaco or HIP binary to test>\n";
}
```
## Building
kernelDB has a dependency on an llvm environment in order to build. For now, the best one to use is the rocm-llvm-dev package. It may not be installed by default so you may need to install it.
Alternatively, you can use the Triton llvm that can typically be found somewhere under here: ~/.triton/llvm. To point the build at a specific llvm install, do the following:
```
cd build
cmake -DLLVM_INSTALL_DIR=/opt/rocm/llvm -DCMAKE_INSTALL_PREFIX=~/.local ..
make && make install
```
The above commands will build libkernelDB64.so and copy it to ${CMAKE_INSTALL_PREFIX}/lib while copying the kerneldb include files to ${CMAKE_INSTALL_PREFIX}/include. If you omit the definition of CMAKE_INSTALL_PREFIX, it defaults to /usr/local.

Building with the Triton llvm will work fine for Triton code objects, but kernelDB will not be able to read HIP binaries. As of this writing, the dwarf support in the Triton llvm implementation chokes on HIP-generated dwarf info. On the other hand, building with rocm-llvm-dev work for _both_ HIP binaries and Triton hsaco files. So it's preferable to build against ROCM llvm if you're able to.
