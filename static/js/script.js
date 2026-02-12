async function lookupActor(num) {
    const actorInput = document.getElementById('actor' + num);
    const statusDiv = document.getElementById('status' + num);
    const actorName = actorInput.value.trim();

    if (!actorName) {
        statusDiv.className = 'actor-status not-found';
        statusDiv.textContent = 'Please enter an actor name.';
        statusDiv.style.display = 'block';
        return;
    }

    statusDiv.className = 'actor-status';
    statusDiv.textContent = 'Searching...';
    statusDiv.style.display = 'block';

    try {
        const response = await fetch('/lookup_actor', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ actor_name: actorName })
        });
        const data = await response.json();

        if (data.found) {
            statusDiv.className = 'actor-status found';
            statusDiv.textContent = '✅ ' + data.actor_name + ' — Star Power: ' + Number(data.score).toLocaleString();
        } else {
            statusDiv.className = 'actor-status not-found';
            statusDiv.textContent = '⚠️ ' + data.message;
        }
    } catch (err) {
        statusDiv.className = 'actor-status not-found';
        statusDiv.textContent = '❌ Error looking up actor.';
    }
}

async function calculateCastPower() {
    const actor1 = document.getElementById('actor1').value.trim();
    const actor2 = document.getElementById('actor2').value.trim();
    const actor3 = document.getElementById('actor3').value.trim();
    const resultDiv = document.getElementById('castPowerResult');

    if (!actor1 && !actor2 && !actor3) {
        resultDiv.style.display = 'block';
        resultDiv.textContent = 'Please enter at least one actor name.';
        resultDiv.style.background = '#fff3cd';
        return;
    }

    resultDiv.style.display = 'block';
    resultDiv.textContent = 'Calculating...';
    resultDiv.style.background = '#e8dff5';

    try {
        const response = await fetch('/calculate_cast_power', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ actor1, actor2, actor3 })
        });
        const data = await response.json();

        if (data.error) {
            resultDiv.textContent = data.error;
            resultDiv.style.background = '#fff3cd';
            return;
        }

        let text = '⚡ Cast Star Power Total: ' + Number(data.total_cast_power).toLocaleString();
        text += '\n';
        data.actors.forEach(function (a, i) {
            const weights = data.actors.length >= 3 ? [0.5, 0.3, 0.2] :
                data.actors.length === 2 ? [0.625, 0.375] : [1.0];
            const status = a.found ? '✅' : '⚠️ (mean)';
            text += '\n' + (i + 1) + '. ' + a.name + ': ' + Number(a.score).toLocaleString() + ' × ' + weights[i] + ' ' + status;
        });

        resultDiv.style.whiteSpace = 'pre-line';
        resultDiv.textContent = text;
    } catch (err) {
        resultDiv.textContent = '❌ Error calculating cast power.';
        resultDiv.style.background = '#f8d7da';
    }
}
