#if !defined( CPlinkMapFile_h )
#define CPlinkMapFile_h
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
 *         File Name:   CPlinkMapFile.h
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

// This file must be #included prior to CPLinkFile.h

#include "Cplink.h"

namespace plink {

	struct MapRecord
	{
		SnpInfo snpInfo;           // the MapRecord only contains SNP description information
	};

	class CPlinkMapFile
	{
	public:
		CPlinkMapFile();
		CPlinkMapFile(const std::string& filename_);
#if 0
		CPlinkMapFile( const std::string& filename_, size_t cIndividuals_, size_t cSnps_ );
#endif
		~CPlinkMapFile();

		void     Load();
		void     Load(const std::string& filename_);

		size_t   CountOfMapRecords() { size_t rc = rgMapRecords.size(); return(rc); }
		size_t   CountOfSnps() { return(CountOfMapRecords()); }
		SnpInfo* PSnpInfo(const std::string& idSnp);
		SnpInfo* PSnpInfo(size_t idxSnpInfo);
		MapRecord* PMapRecord(size_t idx) { MapRecord* pmr = &rgMapRecords[idx]; return(pmr); }

		//private:
		std::string   filename;
#if 0
		FILE     *pFile;
#endif

		std::vector< MapRecord >   rgMapRecords;
		std::map< std::string, size_t > idSnpToSnpInfoIndex;
	};
}//end :plink
#endif      // CPlinkMapFile_h
