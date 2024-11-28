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
        }
    }
    else
        std::cout << "Usage: kdbtest <hsaco or HIP binary to test>\n";
}
