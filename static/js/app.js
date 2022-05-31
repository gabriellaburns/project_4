
console.log("data", data);
// });

d3.select("#selDataset").on("change", optionChanged);

var dropdown = d3.select("#selDataset");


//set national value in dropdown which will fill in the initial charts  
function opening() {
    // Create list of state names to populate the dropdown menu

    for (i = 0; i < data.length; i++) {
        let opener = data[51];
        dropdown.append("option")
            .text(data[i].Name)
            .property("value")
    }
   
   
    createPlots(opener);

}

opening();

function createPlots(sample) {

    let chosenState = data.filter(obj => obj.Name == sample)[0];
    console.log('cs', chosenState)

    // GENERAL CONFIG OPTIONS
    let pielayout = {
        margin: { "t": 0, "b": 0, "l": 0, "r": 0 },
        grid: { rows: 1, columns: 2 },

    };

    let config = { responsive: true, automargin: true };

    let colorSet = [["#DB444B", "#006BA2", "#3EBCD2", "#379A8B", "#EBB434", "#B4BA39", "#9A607F", "#D1B07C"]];

    // REQUEST/APPROVED BAR
    let trace1 = {
        x: [chosenState.Name],
        y: [chosenState.Average_Loan_000s],
        name: "Loan Request (in 000's)",
        type: "bar",
        marker: {
            color: colorSet[0][0]
        },
        text: "Loan Request",
        textposition: 'auto',

    };
    let trace2 = {
        x: [chosenState.Name],
        y: [chosenState.Average_Income_000s],
        name: "Income Amount",
        type: "bar",
        marker: {
            color: colorSet[0][1]
        },
        text: "Income Amount (in 000's)",
        textposition: 'auto',

    };

    let traceData = [trace1, trace2];
    let barLayout = { barmode: 'group', title: 'Average Loan Request & Income (in 000s)', showlegend: false };

    Plotly.newPlot("bar", traceData, barLayout, config);

    // TOTAL LOANS VS QUALIFIED
    let trace3 = {
        x: [chosenState.Name],
        y: [chosenState.Total_Loans],
        name: "Loans Requested",
        type: "bar",
        marker: {
            color: colorSet[0][0]
        },
        text: "Loans Requested",
        textposition: 'auto',

    };
    let trace4 = {
        x: [chosenState.Name],
        y: [chosenState.Qualified],
        name: "Loans Approved)",
        type: "bar",
        marker: {
            color: colorSet[0][3]
        },
        text: "Loans Approved",
        textposition: 'auto',

    };

    let traceData2 = [trace3, trace4];
    let barLayout2 = { barmode: 'group', title: 'Request & Success Count', showlegend: false };

    Plotly.newPlot("bar2", traceData2, barLayout2, config);

    // PIES, LOTS OF PIES

    var data1 = [{
        values: [chosenState.Applicant_Hispanic_or_Latino, chosenState.Applicant_Not_Hispanic_or_Latino],
        labels: ['Hispanic or Latino', 'Not Hispanic or Latino'],
        type: 'pie',
        // ,
        // insidetextorientation: "auto",
        textinfo: "none",
        hoverinfo: "label+percent",
        title: 'Applicant Ethnicity',
        marker: {
            colors: colorSet[0]
        },
        domain: { column: 0 },
    },
    {
        values: [chosenState['Co-Applicant_Hispanic_or_Latino'], chosenState['Co-Applicant_Not_Hispanic_or_Latino'], chosenState['No_co-applicant_(Ethnicity)']],
        labels: ['Hispanic or Latino', 'Not Hispanic or Latino', 'No Co-Applicant'],
        type: 'pie',
        // automargin: true,
        // insidetextorientation: "auto",
        textinfo: "none",
        hoverinfo: "label+percent",
        title: 'Co-Applicant Ethnicity',
        marker: {
            colors: colorSet[0]
        },
        domain: { column: 1 },
    }];

    Plotly.newPlot('pie1', data1, pielayout, config);


    var data2 = [{
        values: [chosenState.Applicant_American_Indian_or_Alaska_Native,
        chosenState.Applicant_Asian,
        chosenState.Applicant_Black_or_African_American,
        chosenState.Applicant_Native_Hawaiian_or_Other_Pacific_Islander,
        chosenState.Applicant_White],
        labels: ['American Indian or Alaska Native', 'Asian', 'Black or African American', 'Native Hawaiian or Other Pacific Islander', 'White'],
        type: 'pie',
        automargin: true,
        insidetextorientation: "auto",
        textinfo: "none",
        hoverinfo: "label+percent",
        title: 'Applicant Race',
        marker: {
            colors: colorSet[0]
        },
        domain: { column: 0 },
    },
    {
        values: [chosenState['Co-Applicant_American_Indian_or_Alaska_Native'],
        chosenState['Co-Applicant_Asian'],
        chosenState['Co-Applicant_Black_or_African_American'],
        chosenState['Co-Applicant_Native_Hawaiian_or_Other_Pacific_Islander'],
        chosenState['Co-Applicant_White'],
        chosenState['No_co-applicant_(Race)']],
        labels: ['American Indian or Alaska Native', 'Asian', 'Black or African American', 'Native Hawaiian or Other Pacific Islander', 'White', 'No Co-Applicant'],
        type: 'pie',
        automargin: true,
        insidetextorientation: "auto",
        textinfo: "none",
        hoverinfo: "label+percent",
        title: 'Co-Applicant Race',
        marker: {
            colors: colorSet[0]
        },
        domain: { column: 1 },
    }];

    Plotly.newPlot('pie2', data2, pielayout, config);


    var data3 = [{
        values: [chosenState.Applicant_Male, chosenState.Applicant_Female],
        labels: ['Male', 'Female'],
        type: 'pie',
        automargin: true,
        insidetextorientation: "auto",
        textinfo: "none",
        hoverinfo: "label+percent",
        title: 'Applicant Sex',
        marker: {
            colors: colorSet[0]
        },
        domain: { column: 0 },
    },
    {
        values: [chosenState['Co-Applicant_Male'], chosenState['Co-Applicant_Female'], chosenState['No_Co-applicant_(Gender)']],
        labels: ['Male', 'Female', 'Co-Applicant'],
        type: 'pie',
        automargin: true,
        insidetextorientation: "auto",
        textinfo: "none",
        hoverinfo: "label+percent",
        title: 'Co-Applicant Sex',
        marker: {
            colors: colorSet[0]
        },
        domain: { column: 1 },
    }];

    Plotly.newPlot('pie3', data3, pielayout, config);
};




// This function is called when a dropdown menu item is selected
function optionChanged() {

    // Grab the new value in the selector, call it update sample
    updateSample = dropdown.property("value");
    console.log('us', updateSample)

    //Send that value to createPlotly which will update the values for the images  
    createPlots(updateSample);
};
