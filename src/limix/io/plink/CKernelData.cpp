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
 * CKernelData - {Kernel Data Class}
 *
 *         File Name:   CKernelData.cpp
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file implements the CKernelData class for FastLmmC
 *
 *    Change History:   

 *
 * Test Files: 
 */

/*
 * Include Files
 */
#include "CKernelData.h"
#include "TsvLexer.h"
/*
 * Externs we need
 */

/*
 * Globals we define
 */

/*
 * The code
 */
namespace plink {

	CKernelData::CKernelData()
	{
		kernelArray = nullptr;
	}

	CKernelData::~CKernelData()
	{
		if (kernelArray)
		{
			delete[] kernelArray;
		}
	}

	void CKernelData::ReadKernelFile(std::string& kernelFile)
	{
		/*
		 * The kernel file we read is a tab separated file with the form
		 *   var NameXxx NameXxy NameXxz ... <eol>
		 *   NameYyy x x ...<eol>
		 *   NameYyz x x ...<eol>
		 *   ... <eol>
		 *   <eof>
		 *
		 * The first row labels the columns
		 * each subsequent row contains the data as an Attribute Name followed by a real value separated by <tab> for each column
		 */
		cRows = 0;
		cColumns = 0;

		CTsvLexer lex(kernelFile);
		CToken tok;

		if ((lex.NextToken(tok) != tokSymbol) || (tok.text != "var"))
		{
			Fatal("Expected the file [%s] to start with 'var' but found '%s'", kernelFile.c_str(), tok.text.c_str());
		}

		lex.NextToken(tok);         // consume the 'var' token

		// Get all the column headers
		while ((tok.type != tokEOF) && (tok.type != tokEOL))
		{
			if (tok.type != tokTab)
			{
				Fatal("Expected <tab> to precede column header %d at Line: %d Column: %d", columnLabels.size(), tok.line, tok.column);
				break;
			}
			if (lex.NextToken(tok) != tokSymbol)
			{
				Fatal("Expected column header at Line: %d, Column: %d\n  Found %s", tok.line, tok.column, tok.text.c_str());
				break;
			}
			else
			{
				if (columnMap.count(tok.text) == 1)
				{
					Fatal("Duplicate column header [%s] found at Line: %d, Column: %d", tok.text.c_str(), tok.line, tok.column);
				}

				columnLabels.push_back(tok.text);
				columnMap[tok.text] = cColumns;
				++cColumns;
				lex.NextToken(tok);                  // consume the column header
			}
		}

		// Validate the column headers are OK as best we can
		if (columnLabels.size() != cColumns)
		{
			Fatal("Program Error.  columnLabels.size()[%d] != cColumns[%d]", columnLabels.size(), cColumns);
		}
		if (cColumns < 1)
		{
			Fatal("Kernel Data must have 1 or more columns of data.");
		}

		Verbose("Found %d columns of Kernel data", columnLabels.size());

		lex.NextToken(tok);         // consume the <eol> and setup for label
		// Now get the rows
		std::vector<limix::mfloat_t> kernelRow;
		while (tok.type != tokEOF)   // for all the rows
		{
			if (tok.type == tokEOL)  // a blank line is acceptable at the end of the file
			{                       //   consume <eol> until <eof>
				while (lex.NextToken(tok) != tokEOF)
				{
					if (tok.type != tokEOL)
					{
						Fatal("Expecting <eof> and found [%s] at Line: %d, Column: %d", tok.text.c_str(), tok.line, tok.column);
					}
				}
				break;                  // was clean <eol>'s to end of file.  Ignore them and continue
			}
			if (tok.type != tokSymbol)
			{
				Fatal("Expected RowID on Line: %d", tok.line);
			}
			if (rowMap.count(tok.text) != 0)
			{
				Fatal("Duplicate row ID [%s] found at Line: %d, Column: %d", tok.text.c_str(), tok.line, tok.column);
			}
			rowLabels.push_back(tok.text);
			rowMap[tok.text] = cRows;
			++cRows;
			lex.NextToken(tok);         // consume the ID token.  Expect <tab>

			kernelRow.clear();

			while ((tok.type != tokEOF) && (tok.type != tokEOL))
			{
				if (tok.type != tokTab)
				{
					Fatal("Expected <tab> after row label on RowID: %s, Line: %d", rowLabels[rowLabels.size() - 1].c_str(), tok.line);
				}
				lex.NextToken(tok);      // kernel id real
				if (tok.type != tokSymbol)
				{
					Fatal("Expected kernel data (0 or 1) for RowID: %s at Line: %d, Column: %d.  Found '%s'.", rowLabels[rowLabels.size() - 1].c_str(), tok.line, tok.column, tok.text.c_str());
				}

				kernelRow.push_back((limix::mfloat_t)atof(tok.text.c_str()));
				lex.NextToken(tok);      // consume the kernel info type token
			}

			// confirm we have all the kernel attributes we are supposed to have
			size_t cKernel = kernelRow.size();
			if (cKernel != columnLabels.size())
			{
				Fatal("Mismatch in Kernel data size on RowID: %s, Line: %d\n  Expected %d elements and found %d elements.", rowLabels[rowLabels.size() - 1].c_str(), tok.line, columnLabels.size(), cKernel);
			}
			kernelVectors.push_back(kernelRow);
			lex.NextToken(tok);   // advance past eol or it stays on eof
		}

		if (cRows < 1)
		{
			Fatal("Kernel Data must have 1 or more rows of data.");
		}
		if (cColumns != cRows)
		{
			//Fatal( "Kernel must be a square matrix.  We have [%d] columns and [%d] rows.", cColumns, cRows );//should be Fatal
			Warn("Kernel must be a square matrix.  We have [%d] columns and [%d] rows.", cColumns, cRows);//should be Fatal
		}

		Verbose("Found %d rows of Kernel data", rowLabels.size());
		FatalIfPreviousErrors();

		if (fCreateReadValidationFiles)
		{
			std::string s = kernelFile + ".validate.txt";
			WriteKernelFile(s);
		}
	}

	void CKernelData::WriteKernelFile(std::string& kernelFile)
	{
		/*
		 * The kernel file we write is a tab separated file with the form
		 *   var NameXxx NameXxy NameXxz ... <eol>
		 *   NameYyy x x ...<eol>
		 *   NameYyz x x ...<eol>
		 *   ... <eol>
		 *   <eof>
		 *
		 * The first row labels the columns
		 * each subsequent row contains the data as an Attribute Name followed by a real value separated by <tab> for each column
		 */
		FILE *pFile;

		if (kernelFile.size() == 0)
		{
			Fatal("No output filename to open ");
		}

		pFile = fopen(kernelFile.c_str(), "wt");          // write ascii

		if (!pFile)
		{
			std::string fullPath = FullPath(kernelFile);
			Fatal("Cannot open output file [%s].  \n  CRT Error %d: %s", fullPath.c_str(), errno, strerror(errno));
		}

		cColumns = columnLabels.size();
		cRows = rowLabels.size();

		unsigned int old_exponent_format = _set_output_format(_TWO_DIGIT_EXPONENT);   // to be compatible, use 2 digit exponents

		// Write the first row (var\t<columnLabels><eol>)
		fprintf(pFile, "var");
		for (size_t i = 0; i < cColumns; ++i)
		{
			fprintf(pFile, "\t%s", columnLabels[i].c_str());
		}
		fprintf(pFile, "\n");

		// write the rest of the rows
		for (size_t i = 0; i < cRows; ++i)
		{
			fprintf(pFile, "%s", rowLabels[i].c_str());
			for (size_t j = 0; j < cColumns; ++j)
			{
				fprintf(pFile, "\t%.15lG", kernelVectors[i][j]);    // get all the precision using g format
			}
			fprintf(pFile, "\n");
		}

		_set_output_format(old_exponent_format);                  // restore the default format behavior
		fclose(pFile);
	}

	void CKernelData::CreateColumnMajorArrayFromVectors()
	{
		// create the appropriate array / memory structure for the snp data.
		//   this creates the array in FORTRAN's and MatLab's column major order
		size_t ckernel = cColumns * cRows;
		kernelArray = new limix::mfloat_t[ckernel];
		for (size_t iRow = 0; iRow < cRows; ++iRow)
		{
			for (size_t iCol = 0; iCol < cColumns; ++iCol)
			{
				kernelArray[(iRow*cColumns) + iCol] = kernelVectors[iRow][iCol];
			}
		}
	}

	void CKernelData::CreateColumnMajorArrayFromVectorsAndMapping(std::vector<std::string> columnLabelOrder)
	{
		// Create a 'Fortran' style column major array but use the columnLabelOrder to order the columns
		if (cColumns < columnLabelOrder.size())
		{
			Fatal("Cannot create KernelData column major array from Vector and Mapping.  The kernel has fewer individuals than there are phenotype values.  "
				"KernelColumns[%d] and mapping[%d]", cColumns, columnLabelOrder.size());
		}
		else if (cColumns > columnLabelOrder.size())
		{
			Verbose("The kernel has more individuals than there are phenotype values. Removing columns from the kernel...  "
				"KernelColumns[%d] and mapping[%d]", cColumns, columnLabelOrder.size());
		}

		std::vector<size_t> rgMapColumns(columnLabelOrder.size());
		for (size_t iCol = 0; iCol < columnLabelOrder.size(); ++iCol)
		{
			if (columnMap.count(columnLabelOrder[iCol]) != 1)
			{
				Fatal("Cannot create column major array from Vector and Mapping.  kernelColumns contains [%i] copies of mapping key [%s] (epected 1 copy)", columnMap.count(columnLabelOrder[iCol]), columnLabelOrder[iCol].c_str());
			}
			rgMapColumns[iCol] = columnMap[columnLabelOrder[iCol]];
		}

		std::vector<size_t> rgMapRows(columnLabelOrder.size());
		for (size_t iRow = 0; iRow < columnLabelOrder.size(); ++iRow)
		{
			if (rowMap.count(columnLabelOrder[iRow]) != 1)
			{
				Fatal("Cannot create column major array from Vector and Mapping.  kernelRows contains [%i] copies of mapping key [%s] (epected 1 copy)", columnMap.count(columnLabelOrder[iRow]), columnLabelOrder[iRow].c_str());
			}
			rgMapRows[iRow] = columnMap[columnLabelOrder[iRow]];
		}

		size_t ckernel = columnLabelOrder.size() * columnLabelOrder.size();
		kernelArray = new limix::mfloat_t[ckernel];
		for (size_t iRow = 0; iRow < columnLabelOrder.size(); ++iRow)
		{
			for (size_t iCol = 0; iCol < columnLabelOrder.size(); ++iCol)
			{
				kernelArray[(iRow*columnLabelOrder.size()) + iCol] = kernelVectors[rgMapRows[iRow]][rgMapColumns[iCol]];
			}
		}
	}
}//end : plink