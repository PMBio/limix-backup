#if !defined( CPlinkBedFile_h )
#define CPlinkBedFile_h
/*
 *******************************************************************
 *
 *    Copyright (c) Microsoft. All rights reserved.
 *    This code is licensed under the Apache License, Version 2.0.
 *    THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
 *    ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
 *    IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
 *    PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
 *
 ******************************************************************
 */

/*
 * CPlinkBedFile - {PLINK Bed File Access Class}
 *
 *         File Name:   CPlinkBedFile.h
 *           Version:   1.00
 *            Author:   
 *     Creation Date:    4 Dec 2011
 *     Revision Date:    4 Dec 2011
 *
 *    Module Purpose:   This file defines the CPlinkBedFile class 
 *                         for FastLmmC
 *
 *                      A .BED file contains compressed binary genotype values for 
 *                         for individuals by SNPs.  
 *
 *    Change History:   
 *
 * Test Files: 
 */

#include <vector>
#include <string>
#include <limits>
#include <iostream>
#include <stdio.h>
#include <math.h> 

typedef unsigned char BYTE;
typedef unsigned long long uint64_t_;

const BYTE bedFileMagic1 = 0x6C;       // 0b01101100 or 'l' (lowercase 'L')
const BYTE bedFileMagic2 = 0x1B;       // 0b00011011 or <esc>

static double unknownOrMissing = std::numeric_limits<double>::quiet_NaN();
static double homozygousPrimaryAllele = 0.0;
static double heterozygousAllele = 1.0;
static double homozygousSecondaryAllele = 2.0;

static double mapBedGenotypeToRealAllele[4] = { 
                  homozygousSecondaryAllele,       // look-up 0
                  unknownOrMissing,                // look-up 1
                  heterozygousAllele,              // look-up 2
                  homozygousPrimaryAllele,         // look-up 3
                  };

enum BedGenotype     // integer representation of genotype values in Plink's binary .BED file
   {
   bedHomozygousMinor = 0
  ,bedMissingGenotype = 1
  ,bedHeterozygous    = 2
  ,bedHomozygousMajor = 3
   };

enum LayoutMode
   {
   RowMajor=0                    // all elements of a row are sequential in memory
  ,ColumnMajor=1                 // all elements of a colomn are sequential in memory
  ,GroupGenotypesByIndividual=0  // all SNP genotypes for a specific individual are seqential in memory
  ,GroupGenotypesBySnp=1         // all Individual's genotypes for a specific SNP are sequential in memory
   };

class CBedFile
   {
public:
   //CBedFile( const string& filename_ );
   CBedFile( const std::string& filename_, size_t cIndividuals_, size_t cSnps_ );
   ~CBedFile();
   int      NextChar();
   size_t   Read( BYTE *pb, size_t cbToRead );
   size_t   ReadLine( BYTE *pb, size_t idx );

//private:
   static const size_t   cbHeader = 3;         // 
   std::string   filename;
   FILE     *pFile;
   
   LayoutMode  layout;        // 0=RowMajor(all snps per individual together);
                              // 1=ColumnMajor(all individuals per SNP together in memory)
   size_t   cIndividuals;
   size_t   cSnps;
   size_t   cbStride;

   void     Init( const std::string& filename_, size_t cIndividuals_, size_t cSnps_ );
   };

#endif      // CPlinkBedFile_h
