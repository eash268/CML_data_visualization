function getClusteringScores() {
	var data = {
		"user_id" : window.location.pathname.split('/')[window.location.pathname.split('/').length-1]
	}
	$.ajax({
		type: "POST",
		url: "/examplePost",
		data: data,
		success: function(res) {
			res = JSON.parse(res);
			document.getElementById('scs_spinner').style.display = "none";
			document.getElementById('tcs_spinner').style.display = "none";
			document.getElementById('scs').innerHTML = res["scs"];
			document.getElementById('tcs').innerHTML = res["tcs"];
		},
	});
}

$(document).ready(function() { 
	getClusteringScores();
});