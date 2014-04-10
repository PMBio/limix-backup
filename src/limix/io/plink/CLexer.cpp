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
 * CLexer - {Simple Lexer functionality}
 *
 *         File Name:   CLexer.cpp [Simple Lexer Class]
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file implements the basic lexer/scanner for FastLmmC
 *
 *    Change History:   
 *
 * Test Files: 
 */

/*
 * Include Files
 */
#include "CLexer.h"
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
	CLexer::CLexer(const std::string& filename)
	{
		m_fileName = filename;

		if (m_fileName.size() == 0)
		{
			Fatal("Could not create CLexer.  Parameter 'filename' is zero length string");
		}

		m_pFile = fopen(m_fileName.c_str(), "rb");          // read in binary to ensure ftell works right

		if (!m_pFile)
		{
			std::string fullPath = FullPath(m_fileName);
			Fatal("Cannot open input file [%s].\n  CRT Error %d: %s", fullPath.c_str(), errno, strerror(errno));
		}

		m_chLook = 0;
		m_line = 1ul;
		m_column = 1ul;
		AdvanceCh();            // this will read the first character into m_chLook
		//   and advance cchInLine 1 with a '\0' in m_chLook
		m_cchInLine = 0;        //   so reset m_cchInLine properly

		m_text.clear();

	}

	CLexer::~CLexer()
	{
		if (m_pFile)
		{
			if (fclose(m_pFile) != 0)
			{
				std::string fullPath = FullPath(m_fileName);
				Fatal("fclose( %s ) failed.\n  CRT Error %d: %s", fullPath.c_str(), errno, strerror(errno));
			}
			m_pFile = nullptr;
		}

		m_fileName.clear();
	}

	void CLexer::AdvanceCh()
	{
		/*
		 * Advancing the character pointer so keep track of the 'file' location in row column too
		 */
		if (m_chLook == '\n')
		{
			m_column = 1;
			m_cchInLine = 0;
			++m_line;
		}
		else if (m_chLook == '\t')
		{
			m_column = ((m_column + 8) & (~(8 - 1))) + 1;
			++m_cchInLine;
		}
		else if ((m_chLook >= ' ') && (m_chLook <= '~'))         // printable?
		{
			++m_cchInLine;
			++m_column;
		}
		else
		{
			++m_cchInLine;
			// don't know what to do with column (eg. \a or 'bell')
		}

	SkipCh:
		m_chLook = fgetc(m_pFile);                             // advance to next charater?
		if (m_chLook == '\r')
		{
			m_column = 1;
			m_cchInLine = 0;
			goto SkipCh;                                          // swallow <cr> and let them become <nl> or '\n'
		}

	}

	void CLexer::SkipToEol()
	{
		do
		{
			AdvanceCh();
		} while ((LookAheadChar() != EOF)
			&& (LookAheadChar() != '\n'));
	}

	void CLexer::SkipWhiteSpace()
	{
		do
		{
			AdvanceCh();
		} while ((LookAheadChar() == ' ')
			|| (LookAheadChar() == '\t'));
	}
}// end :plink