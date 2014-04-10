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
 * CPlinkMapFile - {PLINK Map File Access Class}
 *
 *         File Name:   CPlinkMapFile.cpp
 *           Version:   1.00
 *            Author:   
 *     Creation Date:    4 Dec 2011
 *     Revision Date:    4 Dec 2011
 *
 *    Module Purpose:   This file implements the CPlinkMapFile class 
 *                         for FastLmmC
 *
 *                      A .MAP file contains the following fields related
 *                         to SNPs.  
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

#include "CPlinkMapFile.h"
namespace plink {

	CPlinkMapFile::CPlinkMapFile()
	{
#if 0
		pFile = nullptr;
#endif
	}

	CPlinkMapFile::CPlinkMapFile(const std::string& filename_)
	{
		filename = filename_;
#if 0
		pFile = nullptr;
#endif
	}

	//CPlinkMapFile::CPlinkMapFile( const string& filename_, size_t cIndividuals_, size_t cSnps_ );
	CPlinkMapFile::~CPlinkMapFile()
	{
#if 0
		if (pFile != nullptr)
		{
			fclose(pFile);
			pFile = nullptr;
		}
#endif
	}

	/*
	 *  Load the .MAP file into memory and make the elements available
	 *    for later access
	 */
	void CPlinkMapFile::Load()
	{
		Verbose("                  Loading .MAP file: [%s]", filename.c_str());
		CTimer timer(true);
		MapRecord mr;              // the working storage
		CToken tok;
		CPlinkLexer lex(filename);

		lex.NextToken(tok);
		while (tok.type != tokEOF)        // for each line until EOF
		{
			if (tok.type == tokEOL)        // blank line...or comment line or ???
			{
				lex.NextToken(tok);
				continue;
			}

			lex.ExpectChromosome(tok, mr.snpInfo.idChromosome, mr.snpInfo.iChromosome);
			lex.ExpectId(tok, mr.snpInfo.idSnp, "SNP Id");
			lex.ExpectReal(tok, mr.snpInfo.geneticDistance, "Genetic Distance");
			lex.ExpectInt(tok, mr.snpInfo.basepairPosition, "Basepair Position");
			if (tok.type != tokEOL)
			{
				Fatal("Expecting <EOL> after Basepair Position at line %d:%d.  Found [%s]", tok.line, tok.column, tok.text.c_str());
			}

			lex.NextToken(tok);      // consume the <EOL>
			mr.snpInfo.majorAllele = '\0';
			mr.snpInfo.minorAllele = '\0';

			// Validate we have a unique SNP id and add it to the map
			if (idSnpToSnpInfoIndex.count(mr.snpInfo.idSnp))
			{
				Fatal("Duplicate SNP Id %s found in SNPs %d and %d", mr.snpInfo.idSnp.c_str(), idSnpToSnpInfoIndex[mr.snpInfo.idSnp] + 1, rgMapRecords.size() + 1);
			}

			idSnpToSnpInfoIndex[mr.snpInfo.idSnp] = rgMapRecords.size();
			rgMapRecords.push_back(mr);
		}

		timer.Stop();
		timer.Report(0x02, "     Loading .MAP file elapsed time: %s");
	}

	/*
	 *  Get SnpInfo associated with idSnp
	 */
	SnpInfo* CPlinkMapFile::PSnpInfo(const std::string& idSnp)
	{
		SnpInfo *pSnpInfo = nullptr;

		if (idSnpToSnpInfoIndex.count(idSnp) != 0)
		{
			// idSnp is found in the map
			size_t idxSnp = idSnpToSnpInfoIndex[idSnp];
			pSnpInfo = &(rgMapRecords[idxSnp].snpInfo);
		}
		return(pSnpInfo);
	}

	/*
	 * Get SnpInfo associated with index
	 */
	SnpInfo* CPlinkMapFile::PSnpInfo(size_t idxSnpInfo)
	{
		SnpInfo *pSnpInfo = nullptr;

		if (idxSnpInfo < rgMapRecords.size())
		{
			pSnpInfo = &(rgMapRecords[idxSnpInfo].snpInfo);
		}
		return(pSnpInfo);
	}
}// end :plink