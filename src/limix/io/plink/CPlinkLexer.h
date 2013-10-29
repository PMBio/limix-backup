#if !defined( CPlinkLexer_h )
#define CPlinkLexer_h
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
 * CPlinkLexer - {PLINK File Lexer Class}
 *
 *         File Name:   CPlinkLexer.h
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file declare the CPlinkLexer class for FastLmmC
 *
 *    Change History:   
 *
 */


/*
 * 'Publish' our defines
 */

/*
 * 'Publish' our class declarations / function prototypes
 */
extern int maxChromosomeValue;

class CPlinkLexer : CLexer
   {
public:
   CPlinkLexer( const string& filename );
   ~CPlinkLexer();
   unsigned NextToken( CToken& tok );
   void ExpectInt( CToken& tok, int& output, const char *description );
   void ExpectReal( CToken& tok, real& output, const char *description );
   void ExpectId( CToken& tok, string& output, const char *description );
   void ExpectSex( CToken& tok, int& output );
   void ExpectPhenotype( CToken& tok, real& output, int& outputType );
   void ExpectAllele( CToken& tok, char& output, const char *description );
   void ExpectProbability( CToken& tok, real& output, const char *description );
   void ExpectSnpAlleles( CToken& tok, char& majorAllele, char& minorAllele );
   void ExpectSnpNucleotides( CToken& tok, char& firstNucleotide, char& secondNucleotide );
   void ExpectSnpProbabilities( CToken& tok, SnpProbabilities& output );          // SnpProbabilities is defined in CPlinkFile.h
   void ExpectChromosome( CToken& tok, string& output, int& intOutput  );
   string& Filename() { return( CLexer::Filename() ); }
   };

#endif
