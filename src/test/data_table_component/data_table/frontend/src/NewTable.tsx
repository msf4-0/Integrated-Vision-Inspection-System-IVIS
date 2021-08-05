import React, { ReactNode, ReactElement, useEffect, useState } from "react"
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"
import {
  DataGrid,
  GridRowsProp,
  GridColDef,
  GridValueGetterParams,
  useGridSlotComponentProps,
  GridRowId,
} from "@material-ui/data-grid"
import { createTheme, Theme, createMuiTheme } from "@material-ui/core/styles"
import { createStyles, makeStyles, withStyles } from "@material-ui/styles"
import Pagination from "@material-ui/lab/Pagination"
import PaginationItem from "@material-ui/lab/PaginationItem"

//Define Rows and Columns
const rows: GridRowsProp = [
  { id: 1, lastName: "Snow", firstName: "Jon", age: 35 },
  { id: 2, lastName: "Lannister", firstName: "Cersei", age: 42 },
  { id: 3, lastName: "Lannister", firstName: "Jaime", age: 45 },
  { id: 4, lastName: "Stark", firstName: "Arya", age: 16 },
  { id: 5, lastName: "Targaryen", firstName: "Daenerys", age: null },
  { id: 6, lastName: "Melisandre", firstName: null, age: 150 },
  { id: 7, lastName: "Clifford", firstName: "Ferrara", age: 44 },
  { id: 8, lastName: "Frances", firstName: "Rossini", age: 36 },
  { id: 9, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 10, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 11, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 12, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 13, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 14, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 15, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 16, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 17, lastName: "Roxie", firstName: "Harvey", age: 65 },
]

//Define Columns
const columns: GridColDef[] = [
  {
    field: "id",
    headerName: "ID",
    headerAlign: "center",
    align: "center",
    flex: 20,
    hideSortIcons: true,
    disableColumnMenu: true,
  },
  {
    field: "firstName",
    headerName: "First name",
    headerAlign: "center",
    align: "center",
    flex: 150,
    hideSortIcons: true,
  },
  {
    field: "lastName",
    headerName: "Last name",
    headerAlign: "center",
    align: "center",
    flex: 150,
    hideSortIcons: true,
  },
  {
    field: "age",
    headerName: "Age",
    headerAlign: "center",
    align: "center",
    type: "number",
    hideSortIcons: true,

    flex: 50,
    resizable: true,
  },
  {
    field: "fullName",
    headerName: "Full name",
    description: "This column has a value getter and is not sortable.",
    headerAlign: "center",
    align: "left",
    hideSortIcons: true,
    sortable: false,
    // width: 160,
    flex: 160,
    resizable: true,
    valueGetter: (params: GridValueGetterParams) =>
      `${params.getValue(params.id, "firstName") || ""} ${
        params.getValue(params.id, "lastName") || ""
      }`,
  },
]

function NewTable({ args, disabled, theme }: ComponentProps): ReactElement {
  useEffect(() => Streamlit.setFrameHeight())
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState<number>(5)

  interface ComponentTheme {
    primaryColor: string | undefined
    secondaryBackgroundColor: string | undefined
    textColor: string | undefined
    backgroundColor: string | undefined
    font: string | undefined
  }
  // Declare theme from Streamlit Component
  let myTheme: ComponentTheme = {
    primaryColor: theme?.primaryColor,
    secondaryBackgroundColor: theme?.secondaryBackgroundColor,
    textColor: theme?.textColor,
    backgroundColor: theme?.backgroundColor,
    font: theme?.font,
  }
  console.log(myTheme)
  // TODO
  // const useStyles = makeStyles({
  //   root: {
  //     // color: myTheme.textColor,

  //     background: myTheme.backgroundColor,
  //     color: myTheme.textColor,
  //     fontFamily: myTheme.font,
  //   },
  //   header: {
  //     background: myTheme.secondaryBackgroundColor,
  //     color: myTheme.textColor,
  //   },
  // })
  function customCheckbox() {
    return {
      "& .MuiCheckbox-root svg": {
        width: 16,
        height: 16,
        backgroundColor: "transparent",
        border: `1px solid ${
          myTheme.backgroundColor === "#ffffff" ? "#d9d9d9" : "#98989A"
        }`,
        borderRadius: 2,
      },
      "& .MuiCheckbox-root svg path": {
        display: "none",
      },
      "& .MuiCheckbox-root.Mui-checked:not(.MuiCheckbox-indeterminate) svg": {
        backgroundColor: "#1890ff",
        borderColor: "#1890ff",
      },
      "& .MuiCheckbox-root.Mui-checked .MuiIconButton-label:after": {
        position: "absolute",
        display: "table",
        border: "2px solid #fff",
        borderTop: 0,
        borderLeft: 0,
        transform: "rotate(45deg) translate(-50%,-50%)",
        opacity: 1,
        transition: "all .2s cubic-bezier(.12,.4,.29,1.46) .1s",
        content: '""',
        top: "50%",
        left: "39%",
        width: 5.71428571,
        height: 9.14285714,
      },
      "& .MuiCheckbox-root.MuiCheckbox-indeterminate .MuiIconButton-label:after":
        {
          width: 8,
          height: 8,
          backgroundColor: "#1890ff",
          transform: "none",
          top: "39%",
          border: 0,
        },
    }
  }

  const useStyles = makeStyles({
    root: {
      border: 0,

      color:
        myTheme.backgroundColor === "#ffffff"
          ? "rgba(0,0,0,1)"
          : "rgba(255,255,255,1)",
      fontFamily: [
        myTheme.font,
        "-apple-system",
        "BlinkMacSystemFont",
        '"Segoe UI"',
        "Roboto",
        '"Helvetica Neue"',
        "Arial",
        "sans-serif",
        '"Apple Color Emoji"',
        '"Segoe UI Emoji"',
        '"Segoe UI Symbol"',
      ].join(","),
      WebkitFontSmoothing: "auto",
      letterSpacing: "normal",
      "& .MuiDataGrid-columnsContainer": {
        backgroundColor:
          myTheme.backgroundColor === "#ffffff"
            ? "#fafafa"
            : myTheme.secondaryBackgroundColor, //working
      },
      "& .MuiDataGrid-iconSeparator": {
        display: "none",
      },
      "& .MuiDataGrid-columnHeader, .MuiDataGrid-cell": {
        //working
        borderRight: `1px solid ${
          myTheme.backgroundColor === "#ffffff"
            ? "#f0f0f0"
            : myTheme.secondaryBackgroundColor
        }`,
      },
      // "& .MuiDataGrid-columnsContainer, .MuiDataGrid-cell": {
      //   borderBottom: `1px solid ${
      //     myTheme.backgroundColor === "#ffffff"
      //       ? "#f0f0f0"
      //       : myTheme.secondaryBackgroundColor
      //   }`,
      // },
      // "& .MuiDataGrid-cell": {
      //   color: myTheme.textColor,
      // },
      "& .MuiPagination-root": {
        color: "secondary",
      },
      ...customCheckbox(),
    },
  })

  function CustomPagination() {
    const { state, apiRef } = useGridSlotComponentProps()

    return (
      <Pagination
        color="primary"
        variant="outlined"
        shape="rounded"
        page={state.pagination.page}
        count={state.pagination.pageCount}
        // @ts-expect-error
        renderItem={(props2) => <PaginationItem {...props2} disableRipple />}
        onChange={(event, value) => apiRef.current.setPage(value)}
      />
    )
  }

  const StyledTable = withStyles({
    root: {
      // background: myTheme.backgroundColor,
      color:
        myTheme.backgroundColor === "#FFFFFF"
          ? "rgba(0,0,0,1)"
          : "rgba(255,255,255,1)",
      fontFamily: myTheme.font,
      "& .MuiDataGrid-columnHeaderCheckbox, & .MuiDataGrid-cellCheckbox": {
        padding: 0,
        justifyContent: "center",
        alignItems: "center",
        width: 16,
        height: 16,

        backgroundColor: "transparent",
        borderRight: `1px solid ${
          myTheme.backgroundColor === "#ffffff" ? "#d9d9d9" : "white"
        }`,
        borderRadius: 2,
      },
    },
    "& .MuiDataGrid-selectedRowCount": {
      color: myTheme.backgroundColor === "#ffffff" ? "black" : "white",
      fontFamily: myTheme.font,
    },

    columnHeader: {
      backgroundColor: myTheme.secondaryBackgroundColor,
    },
    row: {
      "& .MuiDataGrid-selectedRowCount": {
        color: myTheme.backgroundColor === "#ffffff" ? "black" : "white",
        fontFamily: myTheme.font,
      },
    },
  })(DataGrid)
  const classes = useStyles()

  const [selectionModel, setSelectionModel] = useState<GridRowId[]>([])

  console.log(selectionModel)
  return (
    <span>
      <div style={{ height: 500, width: "100%" }}>
        <DataGrid
          classes={{ root: classes.root }}
          autoPageSize
          pagination
          // components={{
          //   Pagination: CustomPagination,
          // }}
          page={page}
          pageSize={pageSize}
          onPageChange={(newPage) => setPage(newPage)}
          onPageSizeChange={(newPageSize) => setPageSize(newPageSize)}
          rows={rows}
          columns={columns}
          rowsPerPageOptions={[5, 10, 20]}
          checkboxSelection
          disableSelectionOnClick
          onSelectionModelChange={(newSelectionModel) => {
            setSelectionModel(newSelectionModel)
            Streamlit.setComponentValue(newSelectionModel)
            console.log(newSelectionModel)
          }}
          selectionModel={selectionModel}
          // onSelectionModelChange={(e) => console.log(e.rows)}
          // classes={{ columnHeader: classes.header, root: classes.root }}
        />
      </div>
    </span>
  )
}

export default withStreamlitConnection(NewTable)
