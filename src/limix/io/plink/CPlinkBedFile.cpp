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
 * CPlinkBedFile - {PLINK BED File Access Class}
 *
 *         File Name:   CPlinkBedFile.cpp
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file implements the CPlinkBedFile 
 *                      class for FastLmmC
 *
 *                      A .BED file contains compressed binary genotype 
 *                         values for for individuals by SNPs.  
 *
 *    Change History:   
 *
 * Test Files: 
 */

/*
 * Include Files
 */
#include "CPlinkBedFile.h"
namespace plink {
	CBedFile::CBedFile(const std::string& filename_, size_t cIndividuals_, size_t cSnps_)
	{
		filename = FullPath(filename_);         // save class local copy of expanded filename
		cIndividuals = cIndividuals_;
		cSnps = cSnps_;
		cbStride = 0;

		if (filename.empty())
		{
			Fatal("Could not create BedFile Reader.  Parameter 'filename' is zero length string");
		}

		pFile = fopen(filename.c_str(), "rb");  // read in binary to ensure ftell works right
		if (!pFile)
		{
			Fatal("Cannot open input file [%s].\n  CRT Error %d: %s", filename.c_str(), errno, strerror(errno));
		}

		//  Verify 'magic' number
		unsigned char rd1 = NextChar();
		unsigned char rd2 = NextChar();
		if ((bedFileMagic1 != rd1) || (bedFileMagic2 != rd2))
		{
			Fatal("Invalid BED file [%s]."
				"\n  BED file header is incorrect."
				"\n  Expected magic number of 0x%02x 0x%02x, found 0x%02x 0x%02x",
				filename.c_str(), bedFileMagic1, bedFileMagic2, rd1, rd2);
		}

		// Verify 'mode' is valid
		unsigned char rd3 = NextChar();
		switch (rd3)
		{
		case 0:  // mode = 'IndividualMajor' or RowMajor
			layout = GroupGenotypesByIndividual;   // all SNPs per individual are sequential in memory
			cbStride = (cSnps + 3) / 4;              // 4 genotypes per byte so round up
			break;
		case 1:  // mode = 'SnpMajor' or ColumnMajor
			layout = GroupGenotypesBySnp;          // all individuals per SNP are sequential in memory
			cbStride = (cIndividuals + 3) / 4;       // 4 genotypes per byte so round up
			break;
		default:
			Fatal("Invalid BED file [%s].  BED file header is incorrect.  Expected mode to be 0 or 1, found %d", filename.c_str(), rd3);
			break;
		}
	}

	CBedFile::~CBedFile()
	{
		if (pFile)
		{
			fclose(pFile);
			pFile = nullptr;
		}
	}

	int CBedFile::NextChar()
	{
		int value = fgetc(pFile);
		if (value == EOF)
		{
			Fatal("Invalid BED file [%s].  Encountered EOF before exepected.", filename.c_str());
		}
		return((unsigned char)value);
	}

	size_t CBedFile::Read(BYTE *pb, size_t cbToRead)
	{
		size_t cbRead = fread(pb, 1, cbToRead, pFile);
		if (cbRead != cbToRead)
		{
			if (feof(pFile))
			{
				Fatal("Encountered EOF before exepected in BED file. Invalid BED file [%s]", filename.c_str());
			}
			int err = ferror(pFile);
			if (err)
			{
				Fatal("Encountered a file error %d in BED file [%s]", err, filename.c_str());
			}
		}
		return(cbRead);
	}

	size_t CBedFile::ReadLine(BYTE *pb, size_t idx)
	{
		long long fpos = cbHeader + (idx*cbStride);
		long long fposCur = _ftelli64(pFile);
		if (fpos != fposCur)
		{
			_fseeki64(pFile, fpos, SEEK_SET);
		}

		size_t cbRead = Read(pb, cbStride);
		return(cbRead);
	}
}// end :plink