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
 * CPhenData - {Phenotype Data Class}
 *
 *         File Name:   CPhenData.cpp
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file implements the CPhenData class for FastLmmC
 *
 *    Change History:   

 *
 * Test Files: 
 */

/*
 * Include Files
 */
#include "CPhenData.h"

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
	CPhenData::CPhenData()
	{
		phenArray = nullptr;
	}

	CPhenData::~CPhenData()
	{
		if (phenArray)
		{
			delete[] phenArray;
		}
	}

	void CPhenData::ReadPhenFile(string& phenFile)
	{
		/*
		 * The phen file is a tab separated file with the form
		 *   var NameXxx NameXxy NameXxz ... <eol>
		 *   NameYyy x x ...<eol>
		 *   NameYyz x x ...<eol>
		 *   ... <eol>
		 *   <eof>
		 *
		 * The first row labels the columns
		 * each subsequent row contains the data as an Attribute Name followed by a 0 or 1 value separated by <tab> for each column
		 */
		cRows = 0;
		cColumns = 0;

		CTsvLexer lex(phenFile);
		CToken tok;

		if ((lex.NextToken(tok) != tokSymbol) || (tok.text != "var"))
		{
			Fatal("Expected the file [%s] to start with 'var' but found '%s'", phenFile.c_str(), tok.text.c_str());
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
		if (columnLabels.size() < 1)
		{
			Fatal("Phenotype Data must have 1 or more columns of data.");
		}

		Verbose("Found %d columns of Phenotype data", columnLabels.size());

		lex.NextToken(tok);         // cosume the <eol> and get the 'row' data 
		std::vector<real> phenRow;
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

			phenRow.clear();              // accumlate a list of phenotype reals

			while ((tok.type != tokEOF) && (tok.type != tokEOL))
			{
				if (tok.type != tokTab)
				{
					Fatal("Expected <tab> after row label on RowID: %s, Line: %d", rowLabels[rowLabels.size() - 1].c_str(), tok.line);
				}

				lex.NextToken(tok);      // consume the <tab> token.  Expect phenotype (0 or 1)
				if ((tok.type != tokSymbol) || (tok.text.size() != 1)
					|| ((tok.text != "0") && (tok.text != "1")))
				{
					Fatal("Expected phen data (0 or 1) for RowID: %s at Line: %d, Column: %d.  Found '%s'.", rowLabels[rowLabels.size() - 1].c_str(), tok.line, tok.column, tok.text.c_str());
				}

				phenRow.push_back((real)atof(tok.text.c_str()));
				lex.NextToken(tok);      // consume the phenotype token
			}

			// confirm we have all the phenotype attributes we are supposed to have
			size_t cPhen = phenRow.size();
			if (cPhen != columnLabels.size())
			{
				Fatal("Mismatch in Phenotype data size on RowID: %s, Line: %d\n  Expected %d elements and found %d elements.", rowLabels[rowLabels.size() - 1].c_str(), tok.line, columnLabels.size(), cPhen);
			}
			phenVectors.push_back(phenRow);
			lex.NextToken(tok);               // advance past <eol> token
		}

		if (rowLabels.size() < 1)
		{
			Fatal("Phen Data must have 1 or more rows of data.");
		}

		Verbose("Found %d rows of Phenotype data", rowLabels.size());
		FatalIfPreviousErrors();

		if (fCreateReadValidationFiles)
		{
			string s = phenFile + ".validate.txt";
			WritePhenFile(s);
		}
	}


	void CPhenData::WritePhenFile(string& phenFile)
	{
		/*
		 * The phen file we write is a tab separated file with the form
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

		if (phenFile.size() == 0)
		{
			Fatal("No output filename to open ");
		}

		pFile = fopen(phenFile.c_str(), "wt");          // write ascii

		if (!pFile)
		{
			string fullPath = FullPath(phenFile);
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
				fprintf(pFile, "\t%.15lG", phenVectors[i][j]);    // get all the precision using g format
			}
			fprintf(pFile, "\n");
		}

		_set_output_format(old_exponent_format);                  // restore the default format behavior
		fclose(pFile);
	}

	void CPhenData::CreateColumnMajorArrayFromVectors()
	{
		// Create a 'Fortran' style column major array from the STL vectors<>
		size_t cphen = cColumns * cRows;
		phenArray = new real[cphen];
		for (size_t iRow = 0; iRow < cRows; ++iRow)
		{
			for (size_t iCol = 0; iCol < cColumns; ++iCol)
			{
				phenArray[(iRow*cColumns) + iCol] = phenVectors[iRow][iCol];
			}
		}
	}

	void CPhenData::CreateColumnMajorArrayFromVectorsAndMapping(std::vector<string> columnLabelOrder)
	{
		// Create a 'Fortran' style column major array but use the columnLabelOrder to order the columns
		if (cColumns != columnLabelOrder.size())
		{
			Fatal("Cannot create PhenData column major array from Vector and Mapping.  Vectors are different lengths.  "
				"PhenColumns[%d] and mapping[%d]", cColumns, columnLabelOrder.size());
		}
		std::vector<size_t> rgMap(cColumns);
		for (size_t iCol = 0; iCol < cColumns; ++iCol)
		{
			if (columnMap.count(columnLabelOrder[iCol]) != 1)
			{
				Fatal("Cannot create column major array from Vector and Mapping.  PhenColumns does not contain mapping key [%s]", columnLabelOrder[iCol].c_str());
			}
			rgMap[iCol] = columnMap[columnLabelOrder[iCol]];
		}

		size_t cphen = cColumns * cRows;
		phenArray = new real[cphen];
		for (size_t iRow = 0; iRow < cRows; ++iRow)
		{
			for (size_t iCol = 0; iCol < cColumns; ++iCol)
			{
				if (rgMap[iCol] != iCol)
				{
#if defined( _MSC_VER )    // Windows/VC uses a %Iu specifier for size_t
					const char *szFmt1 = "\n -- assign phenarray[%Iu]= phenVector[%Iu][%Iu]";
#else                      // Linux/g++ uses a %zu specifier for size_t
					const char *szFmt1 = "\n -- assign phenarray[%zu]= phenVector[%zu][%zu]";
#endif
					fprintf(stderr, szFmt1, (iRow*cColumns) + iCol, iRow, rgMap[iCol]);
				}
				phenArray[(iRow*cColumns) + iCol] = phenVectors[iRow][rgMap[iCol]];
			}
		}
	}

}//end :plink