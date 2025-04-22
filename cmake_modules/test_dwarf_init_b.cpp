extern "C"
{
#include "dwarf.h"
#include "libdwarf.h"
}

int main() {
    Dwarf_Debug dbg;
    Dwarf_Error err;
    return dwarf_init_b(0, DW_DLC_READ, DW_GROUPNUMBER_ANY, NULL, NULL, &dbg, &err);
}