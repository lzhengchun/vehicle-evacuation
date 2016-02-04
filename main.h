#ifndef _MAIN_H_
#define _MAIN_H_

/*
* define the cell data structure 
* array of struct or struct of array?
*/
struct cell{
    float p_TL[4];
    float p_BR[4];
    float capacity;
    float cnt_BR;
    float cnt_TL;
    float speed;
};
struct cell_data_structure
{
    // turn probability when come from left(horizontal) or top(vertical), size: N*4
    unsigned char ** p_TL;  
    // turn probability when come from right(horizontal) or bottom(vertical), size: N*4
    unsigned char ** p_BR; 
    // capacity of the cell
    float * capacity;
    // number of vihicles come from left(horizontal) or top(vertical)
    float * cnt_TL;
    // number of vihicles come from right(horizontal) or bottom(vertical)
    float * cnt_BR;
    // type of the cell, -1 -> horizontal; 1 -> vertical
    char * cell_type;
    unsigned char speed;
};
typedef struct cell_data_structure CELL_DT;

#endif