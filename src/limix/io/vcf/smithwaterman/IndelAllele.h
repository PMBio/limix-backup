#ifndef __INDEL_ALLELE_H
#define __INDEL_ALLELE_H

#include <string>
#include <iostream>
#include <sstream>

using namespace std;

class IndelAllele {
    friend ostream& operator<<(ostream&, const IndelAllele&);
    friend bool operator==(const IndelAllele&, const IndelAllele&);
    friend bool operator!=(const IndelAllele&, const IndelAllele&);
    friend bool operator<(const IndelAllele&, const IndelAllele&);
public:
    bool insertion;
    int length;
    int referenceLength(void);
    int readLength(void);
    int position;
    int readPosition;
    string sequence;

    bool homopolymer(void);

    IndelAllele(bool i, int l, int p, int rp, string s)
        : insertion(i), length(l), position(p), readPosition(rp), sequence(s)
    { }
};

bool homopolymer(string sequence);
ostream& operator<<(ostream& out, const IndelAllele& indel);
bool operator==(const IndelAllele& a, const IndelAllele& b);
bool operator!=(const IndelAllele& a, const IndelAllele& b);
bool operator<(const IndelAllele& a, const IndelAllele& b);

#endif
