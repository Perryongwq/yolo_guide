(function($) {
  showSwal = function(type, msg) {
    'use strict';
    if (type === 'basic') {
      Swal.fire({
        text: msg,
        button: {
          text: "OK",
          value: true,
          visible: true,
          className: "btn btn-primary"
        }
      }).then(function(){
        location.reload();
    });

    } 
  else if (type === 'message') {
      Swal.fire({
        text: msg,
        button: {
          text: "OK",
          value: true,
          visible: true,
          className: "btn btn-primary"
        }
      })
    } 
else if (type === 'not_allow_with_refresh') {
    Swal.fire({
          title: 'Error!',
          text: msg,
          confirmButtonText: 'OK',
          didClose: () => {
            refresh(); 
          }
        });

    }

 else if (type === 'notallow') {
      Swal.fire({
        text: msg,
        button: {
          text: "OK",
          value: true,
          visible: true,
          className: "btn btn-primary"
        }
      })
    } 

 
else if (type === 'error') {
      Swal.fire({
        title: 'Error!',
        text: msg,
        button: {
          text: "OK",
          value: true,
          visible: true,
          className: "btn btn-primary"
        }
      }).then(function(){
        location.reload();
    });

    } 

else if (type === 'title-and-text') {
      Swal.fire({
        title: 'Error!',
        text: msg,
        button: {
          text: "OK",
          value: true,
          visible: true,
          className: "btn btn-primary"
        }
      }).then(function(){
        location.reload();
    });

    } 
    else if (type === 'block-submit') {
      Swal.fire({
        title: 'Not able to submit!',
        text: msg,
        button: {
          text: "OK",
          value: true,
          visible: true,
          className: "btn btn-primary"
        }
      })

    }
    else if (type === 'notification') {
      Swal.fire({
        title: 'Notification!',
        text: msg,
        button: {
          text: "OK",
          value: true,
          visible: true,
          className: "btn btn-primary"
        }
      })

    }
    else if (type === 'edit')  {
      Swal.fire({
        title: "Are you sure to enable editing mode?",
        text: msg,
        icon: "warning",
        showCancelButton: true,
        confirmButtonColor: "#3085d6",
        cancelButtonColor: "#d33",
        confirmButtonText: "Yes, enable it!"
      }).then((result) => {
        if (result.isConfirmed) {
          /*Swal.fire({
            title: "Done!",
            text: "You can edit it now.",
            icon: "success"
          });*/
          //document.querySelector("#main-panel").remove();
          //document.querySelectorAll('.inputfield').forEach(e => e.remove());
          //drawTrafficField(trafficId.toString(), trafficfields,endpoint);
          enableinputfield();
        }
      })
    } 
    else if (type === 'update')  {
      Swal.fire({
        title: "Are you sure to enable editing mode?",
        text: msg,
        icon: "warning",
        showCancelButton: true,
        confirmButtonColor: "#3085d6",
        cancelButtonColor: "#d33",
        confirmButtonText: "Yes, enable it!"
      }).then((result) => {
        if (result.isConfirmed) {

          enableupdateinputfield();
        }
      })
    } 
    else if (type === 'success') {
      Swal.fire({
        title: 'Congratulations!!!!',
        text: msg,
        icon: 'success',
        button: {
          text: "Continue",
          value: true,
          visible: true,
          className: "btn btn-primary"
        }
      })

    }
    else if (type === 'success-message') {
      Swal.fire({
        title: 'Congratulations!',
        text: msg,
        icon: 'success',
        button: {
          text: "Continue",
          value: true,
          visible: true,
          className: "btn btn-primary"
        }
      }).then(function(){
        location.reload();
    });

    } else if (type === 'auto-close') {
   let timerInterval;
Swal.fire({
  title: "Auto close notice!",
  html: msg,
  timer: 2000,
  timerProgressBar: true,
  didOpen: () => {
    Swal.showLoading();
    const timer = Swal.getPopup().querySelector("b");
    timerInterval = setInterval(() => {
      timer.textContent = `${Swal.getTimerLeft()}`;
    }, 100);
  },
  willClose: () => {
    clearInterval(timerInterval);
  }
}).then((result) => {
  /* Read more about handling dismissals below */
  if (result.dismiss === Swal.DismissReason.timer) {
    console.log("I was closed by the timer");
  }
});
    } else if (type === 'warning') {
      Swal.fire({
       title: "Are you sure?",
       text: msg,
       icon: "warning",
       showCancelButton: true,
       confirmButtonColor: "#3085d6",
       cancelButtonColor: "#d33",
       confirmButtonText: "Yes"
    }).then((result) => {
      if (result.isConfirmed) {
  		$("#reasondiv").css({"display":"block"});
            	$("#otmodal").css({"display":"block"});
		$("#scheduletypediv").css({"display":"none"});
		$("#process-container").css({"display":"block"});
       }
      });

    } else if (type === 'custom-html') {
      Swal.fire({
        content: {
          element: "input",
          attributes: {
            placeholder: "Type your password",
            type: "password",
            class: 'form-control'
          },
        },
        button: {
          text: "OK",
          value: true,
          visible: true,
          className: "btn btn-primary"
        }
      })
    }
  }

})(jQuery);