#if !defined( CPlinkDatFile_h )
#define CPlinkDatFile_h
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
 * CPlinkDatFile - {PLINK Dat File Access Class}
 *
 *         File Name:   CPlinkDatFile.h
 *           Version:   1.00
 *            Author:   
 *     Creation Date:    4 Dec 2011
 *     Revision Date:    4 Dec 2011
 *
 *    Module Purpose:   This file implements the CPlinkDatFile class 
 *                         for FastLmmC
 *
 *                      A .DAT file contains the following fields related
 *                      <optional header> SNP A1 A2 F1 I1 F2 I2 </optional header>
 *                      idSnp A C 0.98 0.02 1.00 0.00 <... ...>
 *
 *                            chromosome (1-22, X, Y or 0 if unplaced)
 *                            rs# or snp identifier
 *                            Genetic distance (morgans)
 *                            Base-pair position (bp units)
 *                      This is all represented by a SnpInfo struct
 *
 *    Change History:   
 *
 * Test Files: 
 */

struct IndividualId
   {
   string idFamily;
   string idIndividual;
   };

struct DatRecord                       // information from .DAT file
   {
   string         idSnp;            // rs# or SNP identifier
   char           majorAllele;
   char           minorAllele;
   vector<SnpProbabilities> rgSnpProbabilities;
   };

class CPlinkDatFile
   {
public:
   CPlinkDatFile( const string& filename_ );
   ~CPlinkDatFile();

   void     Load();
   void     Load( vector<FamRecord>* prgFam_ );

   size_t   size() { size_t rc = rgDat.size(); return( rc ); }
   size_t   CountOfSnps() { size_t rc = rgDat.size(); return( rc ); }
   size_t   CountOfIndividuals() { size_t rc = (rgDat.size() > 0) ? rgDat[0].rgSnpProbabilities.size() : 0; return( rc ); }
   size_t   CountOfIndividualIds() { size_t rc = rgIndividualIds.size(); return( rc ); }
   bool     FGetSnpInfo( const string& idSnp, SnpInfo& snpInfo_ );   
   bool     FGetSnpInfo( size_t idxSnp, SnpInfo& snpInfo_ );
   bool     FSnpHasVariation( const string& idSnp );
   bool     FSnpHasVariation( size_t idxSNp );
   DatRecord* DatRecordPointer( const string& idSnp );
   DatRecord* DatRecordPointer( size_t idxSnp );

   SnpProbabilities GetSnpProbabilities( const string& idSnp, const string& idIndividual, const string& idFamily );
   SnpProbabilities GetSnpProbabilities( const string& idSnp, const string& keyIndividual );
   SnpProbabilities GetSnpProbabilities( size_t idxSnp, size_t idxIndividual );

   bool     FIdSnpInDatFile( const string& idSnp );
   bool     FKeyIndividualInDatFile( const string& keyIndividual );
   size_t   IdxFromKeyIndividual( const string& keyIndividual );

//private:
   string   filename;
   FILE     *pFile;

   vector< IndividualId > rgIndividualIds;
   vector< DatRecord >   rgDat;

   map< string, size_t > idSnpToDatRecordIndex;
   map< string, size_t > keyIndividualToDatRecordIndividualIndex;

   size_t IdxFromIdSnp( const string& idSnp );
   size_t IdxFromIdFamilyAndIdIndividual( const string& idFamily, const string& idIndividual );
   };

#endif   // CPlinkDatFile_h
