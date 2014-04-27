namespace limix {
//CMemDataFram

%ignore ARDataFrame::getMatrix;
%ignore ARDataFrame::getColHeader;
%ignore ARDataFrame::getRowHeader;

%rename(getMatrix) ARDataFrame::agetMatrix;
%rename(getRowHeader) ARDataFrame::aetRowHeader;
%rename(getColHeader) ARDataFrame::aetColHeader;
}

//raw include
%include "limix/io/dataframe.h"

namespace limix{

//CRMemDataFrameXd
%template(CRMemDataFrameXd) CRMemDataFrame<MatrixXd>;
%template(CRWMemDataFrameXd) CRWMemDataFrame<MatrixXd>;
%template(ARDataFrameXd) ARDataFrame< MatrixXd >;
%template(AWDataFrameXd) AWDataFrame< MatrixXd >;

}